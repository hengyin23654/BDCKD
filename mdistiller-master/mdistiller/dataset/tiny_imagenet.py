import os
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (
    extract_archive,
    check_integrity,
    download_url,
    verify_str_arg,
)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image
import torch

from typing import Optional, Callable

data_folder = "../../data/"


class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

    Args:
        root (string): Root directory of the dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in a PIL image
           and returns a transformed version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the
           target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
           puts it in root directory. If dataset is already downloaded, it is not
           downloaded again.
        is_sample (bool, optional): If true, samples contrastive examples for the train split.
        k (int, optional): Number of negative samples for contrastive learning.
    """

    base_folder = "tiny-imagenet-200/"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    filename = "tiny-imagenet-200.zip"
    md5 = "90528d7ca1a48142e341f4ef8d21d0de"

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
        is_sample: bool = False,
        k: int = 4096,
    ):
        super(TinyImageNet, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(
            split,
            "split",
            (
                "train",
                "val",
            ),
        )
        self.is_sample = (
            is_sample if split == "train" else False
        )  # Only apply sampling for training split
        self.k = k

        if self._check_integrity():
            print("Files already downloaded and verified.")
        elif download:
            self._download()
        else:
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )
        if not os.path.isdir(self.dataset_path):
            print("Extracting...")
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, "wnids.txt"))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)
        self.targets = [each[1] for each in self.data]
        self.inputs = [each[0] for each in self.data]
        self.classes = list(range(200))

        if self.is_sample:
            self._initialize_sampling()

    def _initialize_sampling(self):
        num_classes = len(self.classes)
        num_samples = len(self.data)
        label = np.zeros(num_samples, dtype=np.int32)

        for i in range(num_samples):
            _, target = self.data[i]
            label[i] = target

        self.cls_positive = [[] for _ in range(num_classes)]
        for i in range(num_samples):
            self.cls_positive[label[i]].append(i)

        self.cls_negative = [[] for _ in range(num_classes)]
        for i in range(num_classes):
            for j in range(num_classes):
                if j == i:
                    continue
                self.cls_negative[i].extend(self.cls_positive[j])

        self.cls_positive = [
            np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)
        ]
        self.cls_negative = [
            np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)
        ]

        print("Dataset initialized with contrastive sampling!")

    def _download(self):
        print("Downloading...")
        download_url(self.url, root=self.root, filename=self.filename)
        print("Extracting...")
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index: int):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.is_sample:
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return image, target, index, sample_idx
        else:
            if self.split == "train":
                return image, target, index
            else:
                return image, target

    def __len__(self):
        return len(self.data)

    def get_path(self, idx: int):
        return self.data[idx][0]


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == "train":
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, "images")
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, "images")
        imgs_annotations = os.path.join(dir_path, "val_annotations.txt")

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split("\t"), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images


def get_tinyimagenet_dataloader(
    batch_size,
    val_batch_size,
    num_workers,
    mean=[0.4802, 0.4481, 0.3975],
    std=[0.2770, 0.2691, 0.2821],
):
    # data_folder = './data'
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    # train_folder = os.path.join(data_folder, 'train')
    # print(data_folder)
    train_set = TinyImageNet(data_folder, transform=train_transform)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_set = TinyImageNet(
        root=data_folder,
        download=True,
        split="val",
        transform=test_transform,
        target_transform=None,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(val_batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )

    return train_loader, test_loader, num_data
