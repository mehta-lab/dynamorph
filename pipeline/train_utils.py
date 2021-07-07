import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable
from typing import Tuple
from scipy import ndimage

def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 - 0.5
    o_y = float(y) / 2 - 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=1, col_axis=2, channel_axis=0,
                           fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.
    # Arguments
        x: 3D numpy array - a 2D image with one or more channels.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows (aka Y axis) in the input image.
                  Direction: left to right.
        col_axis: Index of axis for columns (aka X axis) in the input image.
                  Direction: top to bottom.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation
    # Returns
        The transformed version of the input.
    """
    # Input sanity checks:
    # 1. x must 2D image with one or more channels (i.e., a 3D tensor)
    # 2. channels must be either first or last dimension
    if np.unique([row_axis, col_axis, channel_axis]).size != 3:
        raise ValueError("'row_axis', 'col_axis', and 'channel_axis'"
                         " must be distinct")

    # TODO: shall we support negative indices?
    valid_indices = set([0, 1, 2])
    actual_indices = set([row_axis, col_axis, channel_axis])
    if actual_indices != valid_indices:
        raise ValueError(
            f"Invalid axis' indices: {actual_indices - valid_indices}")

    if x.ndim != 3:
        raise ValueError("Input arrays must be multi-channel 2D images.")
    if channel_axis not in [0, 2]:
        raise ValueError("Channels are allowed and the first and last dimensions.")

    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)

        # Matrix construction assumes that coordinates are x, y (in that order).
        # However, regular numpy arrays use y,x (aka i,j) indexing.
        # Possible solution is:
        #   1. Swap the x and y axes.
        #   2. Apply transform.
        #   3. Swap the x and y axes again to restore image-like data ordering.
        # Mathematically, it is equivalent to the following transformation:
        # M' = PMP, where P is the permutation matrix, M is the original
        # transformation matrix.
        if col_axis > row_axis:
            transform_matrix[:, [0, 1]] = transform_matrix[:, [1, 0]]
            transform_matrix[[0, 1]] = transform_matrix[[1, 0]]
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience. Adapted from
    https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class TripletDataset(Dataset):
    """TripletDataset
    Adapted from https://github.com/TowardHumanizedInteraction/TripletTorch
    The TripletDataset extends the standard Dataset provided by the pytorch
    utils. It provides simple access to data with the possibility of returning
    more than one sample per index based on the label.
    Attributes
    ----------
    labels  : np.ndarray
              Array containing all the labels respectively to each data sample.
              Labels needs to provide a way to access a sample label by index.
    data_fn : Callable
              The data_fn provides access to sample data given its index in the
              dataset. Providding a function instead of array has been chosen
              for preprocessing and other reasons.
    size    : int
              Size gives the dataset size, number of samples.
    n_sample: int
              The value represents the number of sample per index. The other
              samples will be chosen to be the same label as the selected one. This
              allows to augment the number of possible valid triplet when used
              with a tripelt mining strategy.
    """

    def __init__(
            self: 'TripletDataset',
            labels: np.ndarray,
            data_fn: Callable,
            n_sample: int,
    ) -> None:
        """Init
        Parameters
        ----------
        labels  : np.ndarray
                  Array containing all the labels respectively to each data
                  sample. Labels needs to provide a way to access a sample label
                  by index.
        data_fn : Callable
                  The data_fn provides access to sample data given its index in
                  the dataset. Providding a function instead of array has been
                  chosen for preprocessing and other reasons.
        size    : int
                  Size gives the dataset size, number of samples.
        n_sample: int
                  The value represents the number of sample per index. The other
                  samples will be chosen to be the same as the selected one.
                  This allows to augment the number of possible valid triplet
                  when used with a tripelt mining strategy.
        """
        super(Dataset, self).__init__()
        self.labels = labels
        self.data_fn = data_fn
        self.size = len(labels)
        self.n_sample = n_sample

    def __len__(self: 'TripletDataset') -> int:
        """Len
        Returns
        -------
        size: int
              Returns the size of the dataset, number of samples.
        """
        return self.size

    def __getitem__(self: 'TripletDataset', index: int) -> Tuple[np.ndarray]:
        """GetItem
        Parameters
        ----------
        index: int
               Index of the sample to draw. The value should be less than the
               dataset size and positive.
        Returns
        -------
        labels: torch.Tensor
                Returns the labels respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        data  : torch.Tensor
                Returns the data respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        Raises
        ------
        IndexError: If index is negative or greater than the dataset size.
        """
        if not (index >= 0 and index < len(self)):
            raise IndexError(f'Index {index} is out of range [ 0, {len(self)} ]')

        label = np.array([self.labels[index]])
        datum = np.array([self.data_fn(index)])

        if self.n_sample == 1:
            return label, datum

        mask = self.labels == label
        # mask[ index ] = False
        mask = mask.astype(np.float32)

        indexes = mask.nonzero()[0]
        indexes = np.random.choice(indexes, self.n_sample - 1, replace=True)
        data = np.array([self.data_fn(i) for i in indexes])

        labels = np.repeat(label, self.n_sample)
        data = np.concatenate((datum, data), axis=0)

        labels = torch.from_numpy(labels)
        data = torch.from_numpy(data)

        return labels, data


class ImageDataset(Dataset):
    """Basic dataset class without labels for inference
        Attributes
        ----------
        data : np.ndarray
                  The data_fn provides access to sample data given its index in the
                  dataset. Providding a function instead of array has been chosen
                  for preprocessing and other reasons.
        """

    def __init__(
            self: 'ImageDataset',
            data: np.ndarray,
             ) -> None:

        super(Dataset, self).__init__()
        self.data = data
        self.size = len(data)

    def __len__(self: 'ImageDataset') -> int:
        """Len
        Returns
        -------
        size: int
              Returns the size of the dataset, number of samples.
        """
        return self.size

    def __getitem__(self: 'ImageDataset', index: int) -> np.ndarray:
        """GetItem
        Parameters
        ----------
        index: int
               Index of the sample to draw. The value should be less than the
               dataset size and positive.
        Returns
        -------
        labels: torch.Tensor
                Returns the labels respectively to each of the samples drawn.
                First sample is the sample is the one at the selected index,
                and others are selected randomly from the rest of the dataset.
        datum  : torch.Tensor
                sample drawn at the selected index,
        Raises
        ------
        IndexError: If index is negative or greater than the dataset size.
        """
        if not (index >= 0 and index < len(self)):
            raise IndexError(f'Index {index} is out of range [ 0, {len(self)} ]')
        datum = self.data[index]
        return datum


def zscore(input_image, channel_mean=None, channel_std=None):
    """
    Performs z-score normalization. Adds epsilon in denominator for robustness

    :param input_image: input image for intensity normalization
    :return: z score normalized image
    """
    if not channel_mean:
        channel_mean = np.mean(input_image, axis=(0, 2, 3))
    if not channel_std:
        channel_std = np.std(input_image, axis=(0, 2, 3))
    channel_slices = []
    for c in range(len(channel_mean)):
        mean = channel_mean[c]
        std = channel_std[c]
        channel_slice = (input_image[:, c, ...] - mean) / \
                        (std + np.finfo(float).eps)
        # channel_slice = t.clamp(channel_slice, -1, 1)
        channel_slices.append(channel_slice)
    norm_img = np.stack(channel_slices, 1)
    print('channel_mean:', channel_mean)
    print('channel_std:', channel_std)
    return norm_img

def zscore_patch(imgs):
    """
    Performs z-score normalization. Adds epsilon in denominator for robustness

    :param input_image: input image for intensity normalization
    :return: z score normalized image
    """
    means = np.mean(imgs, axis=(2, 3))
    stds = np.std(imgs, axis=(2, 3))
    imgs_norm = []
    for img_chan, channel_mean, channel_std in zip(imgs, means, stds):
        channel_slices = []
        for img, mean, std in zip(img_chan, channel_mean, channel_std):
            channel_slice = (img - mean) / \
                            (std + np.finfo(float).eps)
            # channel_slice = t.clamp(channel_slice, -1, 1)
            channel_slices.append(channel_slice)
        channel_slices = np.stack(channel_slices)
        imgs_norm.append(channel_slices)
    imgs_norm = np.stack(imgs_norm)
    # print('channel_mean:', channel_mean)
    # print('channel_std:', channel_std)
    return imgs_norm