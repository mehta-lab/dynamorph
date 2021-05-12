import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Callable
from typing import Tuple


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
    """Basic dataset class for inference
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
        datum = np.array([self.data[index]])

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