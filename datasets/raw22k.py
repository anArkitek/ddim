import torch
from torch.utils.data import Dataset
from .vision import VisionDataset
import os
import cv2
import numpy as np
from torchvision.transforms import RandomHorizontalFlip


class RAW22K(Dataset):
    """
    
    Args:
        root (string): Root directory for the database files.
        classes (string or list): One of {'train', 'val', 'test'} or a list of
            categories to load. e,g. ['bedroom_train', 'church_outdoor_train'].
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, transform=None, target_transform=None):
        super(RAW22K, self).__init__()
        self.image_paths = []
        for _root, _dirs, _files in os.walk(root):
            self.image_paths.extend([os.path.join(_root, f) for f in _files if f.endswith(".png")])
            
        self.length = len(self.image_paths)
        self.random_flip = RandomHorizontalFlip(p=0.5)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        """
        image_path = self.image_paths[index]
        image = cv2.cvtColor(cv2.imread(image_path,flags=-1), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255
        else:
            raise NotImplementedError
        image = torch.from_numpy(image)
        image = self.random_flip(image)
        return (image, 0)

    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)
