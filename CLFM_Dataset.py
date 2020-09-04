import torch
import torchvision
import numpy as np
from torchvision.datasets.vision import VisionDataset
import os

class CLFM_Dataset(VisionDataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'data'

    def __init__(self, root, train=True, test=False, is_complex=False, transform=None, target_transform=None):
        super(CLFM_Dataset, self).__init__(root)

        self.train = train  # training set or test set
        self.test = test
        self.is_complex = is_complex

        self.data = []
        self.targets = []

        #file_path = os.path.join(self.root, self.base_folder, file_name)

        if self.train:
            # Load test, train, val, data
            with open('data/complex_dataset_10class_train.npy', 'rb') as f:
                self.data = np.load(f)
            with open('data/complex_dataset_10class_train_labels.npy', 'rb') as f:
                self.targets = np.load(f)
        else:
            # Load test, train, val, data
            with open('data/complex_dataset_10class_val.npy', 'rb') as f:
                self.data = np.load(f)
            with open('data/complex_dataset_10class_val_labels.npy', 'rb') as f:
                self.targets = np.load(f)
        if self.test:
            # Load test, train, val, data
            with open('data/complex_dataset_10class_val.npy', 'rb') as f:
                self.data = np.load(f)
            with open('data/complex_dataset_10class_test_labels.npy', 'rb') as f:
                self.targets = np.load(f)

        if self.is_complex:
            self.data = self.data.reshape(self.data.shape[0], -1) # [dataset, signal_len, 2] -> [dataset, signal_len*2]
            self.data = np.expand_dims(self.data, axis=1)
        else:
            self.data = self.data.transpose((0, 2, 1))  # convert to HWC

        class_type = "train" if self.train else "val"
        if self.test: class_type = "test"
        print("Dataset: {} - Data Shape: {}".format(class_type, self.data.shape))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        d, target = self.data[index], self.targets[index]

        return d, target

    def __len__(self):
        return len(self.data)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")