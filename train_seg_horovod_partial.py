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

"""
To make changes:
1. select a source data location
2. select a path to save the model
"""

# ===== Input Data ===========
parent = '/gpfs/CompMicro/rawdata/dragonfly/Bryant/Galina/10-29-2020'
mg_a2 = 'target_binary_A2_mg_patches_2'
mg_b2 = 'target_binary_B2_mg_patches'
mg_c2 = 'target_binary_C2_mg_patches'
glial_b2 = 'patches_glial_twoclass_noaug_fullset'
neuron_b2 = 'patches_neurons_twoclass_noaug_fullset'

experiment = os.path.join(parent, glial_b2)
print(f"\nexperiment set to {experiment}")

# ==== Output Model Path ======
model_path = f'{parent}/glial_model_twoclass_glial_64batch_withaug_noLRreduce'
# Define model
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)

# ==== Parameters =============
batch_size = 64
epochs = 500
base_lr = 0.0001
classes = 3

train_pct = 0.1
val_pct = 0.05
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

    hvd.init()

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    # model params
    args.batch_size = batch_size
    args.epochs = epochs
    args.base_lr = base_lr
    args.classes = classes

    args.train_pct = train_pct
    args.val_pct = val_pct
    args.test_pct = test_pct

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

    # train_X_path = f"/gpfs/CompMicro/rawdata/dragonfly/Bryant/Galina/10-29-2020/" \
    #     f"trainpatches_neuron_model_twoclass_test1/" \
    #     f"train_X/"
    # train_y_path = f"/gpfs/CompMicro/rawdata/dragonfly/Bryant/Galina/10-29-2020/" \
    #     f"trainpatches_neuron_model_twoclass_test1/" \
    #     f"train_y"
    # if not os.path.exists(train_X_path):
    #     os.makedirs(train_X_path, exist_ok=True)
    # if not os.path.exists(train_y_path):
    #     os.makedirs(train_y_path, exist_ok=True)

    # for i, x in enumerate(train_X):
    #     np.save(os.path.join(train_X_path, f"{i}.npy"), x)
    #
    # for i, y in enumerate(train_y):
    #     np.save(os.path.join(train_y_path, f"{i}.npy"), y)
