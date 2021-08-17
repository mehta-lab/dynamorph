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
from NNsegmentation.layers import metricsHistory
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
glial_b2 = 'patches_glial_oneclass_noaug_subset'
neuron_b2 = 'patches_neurons_twoclass_noaug_subset'

experiment = os.path.join(parent, neuron_b2)
print(f"\nexperiment set to {experiment}")

# ==== Output Model Path ======
model_path = f'{parent}/neuron_model_twoclass_noaug_subset_2'
# Define model
if not os.path.exists(model_path):
    os.makedirs(model_path, exist_ok=True)

# ==== Parameters =============
batch_size = 32
epochs = 500
base_lr = 0.00001
classes = 3

train_pct = 0.2
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
    y = y.reshape((-1, args.classes+1,) + (256, 256))

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

    train_fs, val_fs, test_fs = build_train_test_split(experiment,
                                                       args.train_pct,
                                                       args.val_pct,
                                                       args.test_pct)

    print(f"train-test split constructed, num(total, train, val,test): {len(train_fs)}, {len(val_fs)}, {len(test_fs)}")
    print(f"\tloading files ...")
    train_patch_list = load_zarr(train_fs)
    val_patch_list = load_zarr(val_fs)
    test_patch_list = load_zarr(test_fs)

    print(f"\tpreprocessing files ...")
    train_X, train_y = pre(train_patch_list,
                           args.classes,
                           'annotation')
    val_X, val_y = pre(val_patch_list,
                       args.classes,
                       'annotation')

    # create generators
    # train_gen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
    train_gen = ImageDataGenerator()
    val_gen = ImageDataGenerator()

    # initialize model
    model = Segment(input_shape=(2, 256, 256),  # Phase + Retardance
                    unet_feat=32,
                    fc_layers=[64, 32],
                    n_classes=args.classes,  # predict 2 classes and background
                    model_path=model_path)

    model.compile_model(opt=None, lr=0.001)

print(f"fitting data")
# michael runs fit 5 times, first on human annotations, second on combined human+RF results
for st in range(5):
    model.fit(train_patch_list,
              label_input='annotation',
              batch_size=32,
              n_epochs=200,
              valid_patches=val_patch_list,
              valid_label_input='annotation')
    # model.save(model.model_path + '/stage%d.h5' % st)


# for site in TEST_DATA_PATH:
#   print(site)
#   predict_whole_map(TEST_DATA_PATH[site],
#                     model,
#                     n_classes=3,
#                     batch_size=8,
#                     n_supp=5)