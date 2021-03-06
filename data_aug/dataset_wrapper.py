import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageFile
import pandas as pd
from tqdm import tqdm
import sys


np.random.seed(0)
ImageFile.LOAD_TRUNCATED_IMAGES = True


def check_for_kernelsize(ksize):
    if ksize % 2 == 1:
        return ksize
    else:
        return ksize + 1


class DataSetWrapper(object):

    def __init__(self, batch_size, num_workers, valid_size, input_shape, s):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.s = s
        self.input_shape = eval(input_shape)

    def get_data_loaders(self, data_csv=None):
        data_augment = self._get_simclr_pipeline_transform()

        if not data_csv:
            train_dataset = datasets.STL10('./data', split='train+unlabeled', download=True,
                                           transform=SimCLRDataTransform(data_augment))
        else:
            train_dataset = SimpleDataset(data_csv, transforms=SimCLRDataTransform(data_augment))

        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def _get_simclr_pipeline_transform(self):
        # get a set of data augmentation transformations as described in the SimCLR paper.
        color_jitter = transforms.ColorJitter(0.8 * self.s, 0.8 * self.s, 0.8 * self.s, 0.2 * self.s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=self.input_shape[0]),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply([color_jitter], p=0.8),
                                              transforms.RandomGrayscale(p=0.2),
                                              GaussianBlur(kernel_size=check_for_kernelsize(int(0.1 * self.input_shape[0]))),
                                              transforms.ToTensor()])
        return data_transforms

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True, shuffle=False)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)
        return train_loader, valid_loader


class SimCLRDataTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        xi = self.transform(sample)
        xj = self.transform(sample)
        return xi, xj


class SimpleDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.column = "img_path"

        self.data = pd.read_csv(csv_file).to_dict("records")

        # file exist
        self.data = [n for n in tqdm(self.data) if check_image(n[self.column])]
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Assuming images are in image_url
        """
        sample = Image.open(self.data[idx]['img_path'])
        # sample = sample.resize((256, 256))
        # sample = cv2.imread(self.data[idx]['img_path'])
        # sample = cv2.resize(sample, (224, 224)) / 255.0
        # sample = np.asarray(sample, dtype=np.float32)
        sample = self.transforms(sample)

        return sample, "1"


def check_image(img):
    try:
        img = Image.open(img)
        temp = img.size
        if temp[0] > 0 and temp[1] > 0:
            return True
        else:
            raise Exception("error")
    except KeyboardInterrupt:
        sys.exit()
    except:
        return False
