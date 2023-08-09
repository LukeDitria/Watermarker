from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
import scipy.io
from torchvision.datasets import CIFAR10


class CelebADataset(Dataset):
    def __init__(self, dataset_root, transform=transforms.ToTensor()):
        self.dataframe = pd.read_csv(os.path.join(dataset_root, "attributes.csv"))

        filepath_log = []
        fileindx_log = []

        for root, dirs, files in os.walk(dataset_root):
            for file in files:
                if file.split(".")[1] == "png":
                    file_path = os.path.join(root, file)
                    filepath_log.append(file_path)
                    fileindx_log.append(int(file.split(".")[0]))

        self.file_dict = dict(zip(fileindx_log, filepath_log))

        self.transform = transform
        self.dataset_root = dataset_root

    def __getitem__(self, index):
        abs_file_path = self.file_dict[index]
        img = self.transform(Image.open(abs_file_path))

        class_index = ((self.dataframe.iloc[index].to_numpy()[1:] + 1) / 2).astype(np.float)

        return img, torch.tensor(class_index).float()

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