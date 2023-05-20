import torch
from torch.utils.data import Dataset
# from .vision import VisionDataset
import os
import cv2
import numpy as np
import torchvision as tv
from PIL import Image


def resize_single_channel_16bit(x_np, output_size):
    img = Image.fromarray(x_np.astype(np.float32), mode='F')
    img = img.resize(output_size, resample=Image.BICUBIC)
    return np.asarray(img).clip(0, 1).reshape(output_size[0], output_size[1], 1)

def resize_image(image, ratio):
    """
    Args:
        image (np.array): range [0,255] if 8bit; [0,65535] if 16bit; shape (h,w,c)
        ration (float)
    Returns: 
        image (np.array, dtype=np.float32); range [0,1]; shape(h,w,c)
    """
    image_h = image.shape[0]
    image_w = image.shape[1]
    output_size = (int(image_w / ratio), int(image_h / ratio))
    
    if image.dtype == np.uint8:
        image = Image.fromarray(image)
        image = image.resize(output_size)
        image = np.asarray(image, np.float32) / 255 # [0, 255] -> [0, 1]
    elif image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535 # [0, 65535] -> [0, 1]
        image = [resize_single_channel_16bit(image[:, :, idx], output_size) for idx in range(3)]
        image = np.concatenate(image, axis=2).astype(np.float32)
    else:
        raise NotImplementedError
    assert 0.0 <= image.min() and image.max() <= 1.0 
    return image


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

    def __init__(self, root, resolution, transform=None, target_transform=None):
        super(RAW22K, self).__init__()
        self.image_paths = []
        extensions = ["jpg", "png"]
        for _root, _dirs, _files in os.walk(root):
            self.image_paths.extend([os.path.join(_root, f) for f in _files if f.split(".")[-1] in extensions])
            
        self.length = len(self.image_paths)
        self.resolution = resolution
        
        self.transform = tv.transforms.Compose(
            [
                tv.transforms.CenterCrop(resolution),
                tv.transforms.RandomHorizontalFlip(p=0.5)
            ]
        )


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        """
        image_path = self.image_paths[index]
        image = cv2.cvtColor(cv2.imread(image_path, flags=-1), cv2.COLOR_BGR2RGB)
        ratio = min(image.shape[0], image.shape[1]) / self.resolution
        # image = image.resize(int(image_w / ratio), int(image_h / ratio))
        image = resize_image(image, ratio) # [0,1]
        image = image.transpose(2, 0, 1) # [h, w, c] -> [c, h, w]
        image = torch.from_numpy(image)
        image = self.transform(image)
        return (image, 0)

    def __len__(self):
        return self.length

    def extra_repr(self):
        return "Classes: {classes}".format(**self.__dict__)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    dataset = RAW22K(root="/mnt/localssd/datasets/lsun/churches/church_256x256_raws", resolution=256)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    data_iter = iter(dataloader)
    samples = next(data_iter)
    for i in range(len(samples[0])):
        image = samples[0][i].numpy().transpose(1, 2, 0)
        image = (image * 65535).astype(np.uint16)
        cv2.imwrite(f"{i}.png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
