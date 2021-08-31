import numpy as np
import os, sys
os.environ['KERAS_BACKEND'] = 'tensorflow'
# sys.path.append(os.getcwd())
from NNsegmentation.models import Segment
import zarr
from natsort import natsorted
import horovod.tensorflow as hvd
from horovod.tensorflow.keras import callbacks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import argparse
import logging
from datetime import datetime
from NNsegmentation.data import preprocess
from NNsegmentation.layers import metricsHistory, ValidMetrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import segmentation_models as sm

"""
To make changes:
1. select a source data location
2. select a path to save the model
"""

# ===== Input Data ===========
parent = '/gpfs/CompMicro/rawdata/dragonfly/Bryant/Galina/10-29-2020'

# not enough glial samples
glial_B2 = 'patches_glial_oneclass'

# expanded neuron samples
neuron_C2 = 'patches_neurons_oneclass'
neuron_C2_2 = 'patches_neurons_oneclass_2'

# mega set of mg samples
mg_all_wells = 'patches_mg_oneclass'

experiment = os.path.join(parent, 'patches', neuron_C2_2)
print(f"\nexperiment set to {experiment}")

# ==== Output Model Path ======
model_expt = 'neuron_model_oneclass_run2-3'
model_path = os.path.join(parent, 'models', model_expt)
# Define model
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)

# ==== Parameters =============
batch_size = 64
epochs = 200
base_lr = 0.0001
classes = 2

train_pct = 0.85
val_pct = 0.15
test_pct = 0


def build_train_test_split(zarr_files, train_percent, val_percent, test_percent):
    """
    :param zarr_files: full path to folder containing multiple .zarr files
    :param train_percent: percentage to set aside for training
    :param val_percent: percentage to set aside for validation
    :param test_percent: percentage to set aside for test
    :return:
    """
    zs = natsorted(os.listdir(zarr_files))

    # use indicies to determine split
    indicies = set(range(len(zs)))
    val_num = int(val_percent*len(indicies))
    test_num = int(test_percent*len(indicies))
    train_num = int(train_percent*(len(indicies)))

    # select random val
    val = np.random.choice(list(indicies), val_num, replace=False)

    # remove val indicies from set and select random test
    test_indicies = indicies - set(val)
    test = np.random.choice(list(test_indicies), test_num, replace=False)

    # remove test_indicies from set and remaining is train
    train_indicies = indicies - set(val) - set(test)
    train = np.random.choice(list(train_indicies), train_num, replace=False)

    train_fs = [os.path.join(zarr_files, f) for i, f in enumerate(zs) if i in train]
    val_fs = [os.path.join(zarr_files, f) for i, f in enumerate(zs) if i in val]
    test_fs = [os.path.join(zarr_files, f) for i, f in enumerate(zs) if i in test]

    return train_fs, val_fs, test_fs


def load_zarr(files):
    """
    load zarr files, recast as numpy array, split into X, y (phase/retardance, binary mask), return all pairs as list

    :param files: list
        list of full file paths to .zarr image files
        each image file should be of shape (C, Y, X) with C = (binary, phase, retardance)
    :return: list of [X, y]
        each element should be of dims (C, Z, Y, X)
    """
    out = []
    for f in files:
        z = zarr.open(f)
        z_shape = z.shape
        if z_shape[0] == 3:
            arr = np.array(z).reshape((z_shape[0], 1, z_shape[1], z_shape[2]))
            # if t, c, z, y, x
            # X = arr[1:].reshape((1,)+arr[1:].shape)
            # y = arr[0].reshape((1, 1,)+arr[0].shape)

            # if c, z, y, x
            X = arr[1:].reshape(arr[1:].shape)
            y = arr[0].reshape((1,)+arr[0].shape)
        elif z_shape[0] == 4:
            arr = np.array(z).reshape((z_shape[0], 1, z_shape[1], z_shape[2]))
            # if t, c, z, y, x
            # X = arr[1:].reshape((1,)+arr[1:].shape)
            # y = arr[0].reshape((1, 1,)+arr[0].shape)

            # if c, z, y, x
            X = arr[2:]
            y = arr[:2]
        else:
            raise NotImplementedError()

        # arr[1:] is X (phase, ret)
        # arr[0] is y (binary mask)
        out.append([X, y])
    return out


def pre(patches, cls, inp):
    X, y = preprocess(patches,
                      n_classes=cls,
                      label_input=inp,
                      class_weights=None)

    X = X.reshape((-1,) + (2, 256, 256))
    # y = y.reshape((-1, args.classes+1,) + (256, 256))
    y = y.reshape((-1, args.classes,) + (256, 256))

    # X = X.reshape((-1,) + (256, 256, 2))
    # # y = y.reshape((-1, args.classes+1,) + (256, 256))
    # y = y.reshape((-1, ) + (256, 256, args.classes))

    assert X.shape[0] == y.shape[0]
    return X, y


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-o', '--model_output_dir',
        type=str,
        required=False,
        help="path to the directory to write trained models",
    )
    parser.add_argument(
        '-p', '--project_dir',
        type=str,
        required=False,
        help="path to the project folder containing the JUNE/raw subfolder",
    )
    parser.add_argument(
        '-c', '--channels',
        type=lambda s: [int(item.strip(' ').strip("'")) for item in s.split(',')],
        required=False,
        help="list of integers like '1,2,3' corresponding to channel indicies",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # system params
    os.environ['WORLD SIZE'] = '4'
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '631'

    hvd.init()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # args.workers = 1
    # args.no_cuda = False
    # args.fp16_allreduce = False
    # args.batches_per_allreduce = 1
    # args.use_adasum = False
    # args.distributed = True

    # model params
    args.batch_size = batch_size
    args.epochs = epochs
    args.base_lr = base_lr
    args.classes = classes

    args.train_pct = train_pct
    args.val_pct = val_pct
    args.test_pct = test_pct

    # set world_size retrieved from MPI
    # if os.getenv('OMPI_COMM_WORLD_SIZE'):
    #     args.world_size = int(os.getenv('OMPI_COMM_WORLD_SIZE'))

    # log.info(f"WORLD SIZE = {os.getenv('OMPI_COMM_WORLD_SIZE')}")

    # ======= launch main worker  with logging =============

    start = datetime.now()

    # start logs
    logging.basicConfig(
        level=logging.DEBUG,
        # format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{model_path}/{start.strftime('%Y_%m_%d_%H_%M')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    log = logging.getLogger(__name__)
    log.setLevel(20)

    log.info(f"WORLD SIZE = {os.getenv('OMPI_COMM_WORLD_SIZE')}")
    log.info(f"HVD Rank= {hvd.local_rank()}")
    log.info(f"HVD Size= {hvd.local_size()}")

    # ===============================================================================
    print(f"Building dataset")
    # build dataset
    train_fs, val_fs, test_fs = build_train_test_split(experiment,
                                                       args.train_pct,
                                                       args.val_pct,
                                                       args.test_pct)
    print(f"train-test split constructed, num(total, train, val,test): {len(train_fs)}, {len(val_fs)}, {len(test_fs)}")
    print(f"\tloading files ...")
    train_patch_list = load_zarr(train_fs)
    val_patch_list = load_zarr(val_fs)
    test_patch_list = load_zarr(test_fs)

    train_batches = len(train_patch_list) // args.batch_size
    val_batches = len(val_patch_list) // args.batch_size

    print(f"\tpreprocessing files ...")
    train_X, train_y = pre(train_patch_list,
                           args.classes,
                           'annotation')
    val_X, val_y = pre(val_patch_list,
                       args.classes,
                       'annotation')

    # ================================ Augmentation  =============================================================
    # create generators
    # train_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True, rotation_range=10,
    #                                horizontal_flip=True, vertical_flip=True)
    # train_gen = ImageDataGenerator()
    train_gen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, data_format='channels_first',
                                   rotation_range=10, shear_range=10)
    val_gen = ImageDataGenerator()

    # ============================================================================================================

    # initialize model
    model = Segment(input_shape=(2, 256, 256),  # Phase + Retardance
                    unet_feat=32,
                    fc_layers=[64, 32],
                    n_classes=args.classes,  # predict 2 classes and background
                    model_path=model_path)

    # wrap optimizers and compile model
    scaled_lr = args.base_lr*hvd.size()
    opt = keras.optimizers.Adam(
        learning_rate=scaled_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam'
    )
    opt = hvd.DistributedOptimizer(opt)

    model.compile_model(opt=opt)

    # ================================ load model weights  =============================================================

    if hvd.rank() == 0:
        model_weights = os.path.join('/gpfs/CompMicro/rawdata/dragonfly/Bryant/Galina/10-29-2020/models',
                                     'neuron_best_model_8-27-2021',
                                     'weights.epoch-160.loss-0.216.valloss-0.108.hdf5')
        model.load(model_weights)

    # ============================================================================================================

    # assign callbacks
    callbacks = [
        callbacks.BroadcastGlobalVariablesCallback(0),
        callbacks.MetricAverageCallback(),
        callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=5, verbose=1),
        # keras.callbacks.ReduceLROnPlateau(patience=10, verbose=1)
        keras.callbacks.ReduceLROnPlateau()
    ]

    if hvd.rank() == 0:
        callbacks.append(keras.callbacks.ModelCheckpoint(model_path +
                                                         '/weights.epoch-{epoch:03d}.'
                                                         'loss-{loss:.3f}.'
                                                         'valloss-{val_loss:.3f}.'
                                                         'hdf5'))

        run = datetime.now().strftime('%Y_%m_%d_%H_%M')
        callbacks.append(keras.callbacks.TensorBoard(f'/data1/bryant/logs/run_{run}_{model_expt}'))

    history = metricsHistory()
    callbacks.append(history)

    # valid_score_callback = ValidMetrics()
    # valid_score_callback.valid_data = (val_X, val_y)
    # callbacks.append(valid_score_callback)

    # fit data, but don't use the model's wrapped method
    print(f"fitting data")
    for stage in range(1):
        model.model.fit(train_gen.flow(train_X,
                                       train_y,
                                       batch_size=args.batch_size),
                        callbacks=callbacks,
                        epochs=args.epochs,
                        verbose=1,
                        validation_data=val_gen.flow(val_X,
                                                     val_y,
                                                     batch_size=args.batch_size),
                        validation_steps=3*val_batches // hvd.size()
                        )
        # model.save(os.path.join(model_path, f'/stage{stage}.model'))
        if hvd.rank() == 0:
            model.save(f'/data1/bryant/logs/run_{run}_{model_expt}/stage{stage}.model')

    stop = datetime.now()
    log.info(f"================ END vq-vae training ============== ")
    log.info(f"================== {stop.strftime('%Y_%m_%d_%H_%M')} ================= ")
    log.info(f"time elapsed = {(stop-start).days}-days_{(stop-start).seconds//60}-minutes_{(stop-start).seconds%60}-seconds")