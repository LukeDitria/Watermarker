from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import scipy.io
from torchvision.datasets import CIFAR10


class CelebAHQDataset(Dataset):
    def __init__(self, dataset_root, target_attribute, transform=transforms.ToTensor()):
        self.dataframe = pd.read_csv(os.path.join(dataset_root, "attributes.csv"))

        self.transform = transform
        self.dataset_dir_root = os.path.join(dataset_root, "Images_128")
        self.target_attribute_index = list(self.dataframe.keys()[1:]).index(target_attribute)

    def __getitem__(self, index):
        abs_file_path = os.path.join(self.dataset_dir_root, str(index) + ".png")
        img = self.transform(Image.open(abs_file_path))
        attribute_vec = (self.dataframe.iloc[index].to_numpy()[1:]).astype(np.float)

        class_index = attribute_vec[self.target_attribute_index]

        return img, class_index, attribute_vec

    def __len__(self):

        return len(self.dataframe)


class Flower102Dataset(Dataset):
    def __init__(self, dataset_root, transform=transforms.ToTensor()):
        self.labels = scipy.io.loadmat(os.path.join(dataset_root, 'imagelabels.mat'))['labels'][0] - 1
        self.filenames = np.sort(os.listdir(dataset_root + '/jpg'))

        self.transform = transform
        self.dataset_root = dataset_root

    def __getitem__(self, index):
        filename = self.filenames[index]
        abs_file_path = os.path.join(self.dataset_root, 'jpg/') + filename

        img = self.transform(Image.open(abs_file_path))

        class_index = int(self.labels[index])
        return img, class_index

    def __len__(self):

        return len(self.filenames)


class ArtBench10(CIFAR10):

    base_folder = "artbench-10-batches-py"
    url = "https://artbench.eecs.berkeley.edu/files/artbench-10-python.tar.gz"
    filename = "artbench-10-python.tar.gz"
    tgz_md5 = "9df1e998ee026aae36ec60ca7b44960e"
    train_list = [
        ["data_batch_1", "c2e02a78dcea81fe6fead5f1540e542f"],
        ["data_batch_2", "1102a4dcf41d4dd63e20c10691193448"],
        ["data_batch_3", "177fc43579af15ecc80eb506953ec26f"],
        ["data_batch_4", "566b2a02ccfbafa026fbb2bcec856ff6"],
        ["data_batch_5", "faa6a572469542010a1c8a2a9a7bf436"],
    ]

    test_list = [
        ["test_batch", "fa44530c8b8158467e00899609c19e52"],
    ]
    meta = {
        "filename": "meta",
        "key": "styles",
        "md5": "5bdcafa7398aa6b75d569baaec5cd4aa",
    }