from __future__ import print_function, division
import cv2
import os
import torch
import json
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import transforms as T
from PIL import Image
import utils
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

LABEL_ENUM = {"nml": 1, "abn": 1} #currently treating both nml and abn as same GB object


class GbClassDataset(Dataset):
    """ GB region of interest dataset. """
    def __init__(self, img_dir, df, labels, img_transforms=None):
        #with open(json_file, "r") as f:
        #    df = json.load(f)
        d = []
        for k, v in df.items():
            v["filename"] = k
            d.append(v)
        l = {}
        for label in labels:
            key, cls = label.split(",")
            l[key] = int(cls)+1
        self.labels = l
        self.df = d
        self.img_dir = img_dir
        self.transforms = img_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.df[idx]["filename"])
        image = io.imread(img_name)
        #image = Image.open(img_name)
        #image = np.array(image)
        num_objs = len(self.df[idx]["bbs"])
        labels = []
        boxes = []
        for i in range(num_objs):
            bbs = self.df[idx]["bbs"][i]
            if bbs[0] in ['abn', 'nml']:
                #labels.append(LABEL_ENUM[bbs[0]])
                labels.append(self.labels[self.df[idx]["filename"]])
                boxes.append(bbs[1])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image.double(), target, self.df[idx]["filename"]


class GbRoiDatasetNew(Dataset):
    """ GB region of interest dataset. """
    def __init__(self, img_dir, imgs, img_transforms=None):
        #with open(json_file, "r") as f:
        #    df = json.load(f)
        self.df = imgs

class GbRoiDataset(Dataset):
    """ GB region of interest dataset. """
    def __init__(self, img_dir, df, img_transforms=None):
        #with open(json_file, "r") as f:
        #    df = json.load(f)
        d = []
        for k, v in df.items():
            v["filename"] = k
            d.append(v)
        self.df = d
        self.img_dir = img_dir
        self.transforms = img_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.df[idx]["filename"])
        image = io.imread(img_name)
        #image = Image.open(img_name)
        #image = np.array(image)
        num_objs = len(self.df[idx]["bbs"])
        labels = []
        boxes = []
        for i in range(num_objs):
            bbs = self.df[idx]["bbs"][i]
            if bbs[0] in ['abn', 'nml']:
                labels.append(LABEL_ENUM[bbs[0]])
                boxes.append(bbs[1])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image.double(), target


class GbRoiDatasetTest(Dataset):
    """ GB region of interest dataset. """
    def __init__(self, img_dir, df, img_transforms=None):
        #with open(json_file, "r") as f:
        #    df = json.load(f)
        d = []
        for k, v in df.items():
            v["filename"] = k
            d.append(v)
        self.df = d
        self.img_dir = img_dir
        self.transforms = img_transforms

    def __len__(self):
        return len(self.df)

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #     img_name = os.path.join(self.img_dir, self.df[idx]["filename"])
    #     image = io.imread(img_name)
    #     #image = Image.open(img_name)
    #     #image = np.array(image)
    #     num_objs = len(self.df[idx]["bbs"])
    #     labels = []
    #     boxes = []
    #     for i in range(num_objs):
    #         bbs = self.df[idx]["bbs"][i]
    #         if bbs[0] in ['abn', 'nml']:
    #             labels.append(LABEL_ENUM[bbs[0]])
    #             boxes.append(bbs[1])
    #     boxes = torch.as_tensor(boxes, dtype=torch.float32)
    #     labels = torch.as_tensor(labels, dtype=torch.int64)
    #     image_id = torch.tensor([idx])
    #     area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    #     # suppose all instances are not crowd
    #     iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
    #     target = {}
    #     target["boxes"] = boxes
    #     target["labels"] = labels
    #     target["image_id"] = image_id
    #     target["area"] = area
    #     target["iscrowd"] = iscrowd
    #     if self.transforms is not None:
    #         image, target = self.transforms(image, target)
    #     return image.double(), target, self.df[idx]["filename"]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.df[idx]["filename"])
        image = io.imread(img_name)
        
        # Get the bounding boxes and labels from the correct JSON keys
        num_objs = len(self.df[idx]["boxes"])  # Use "boxes" instead of "bbs"
        labels = []
        boxes = []

        for i in range(num_objs):
            bbs = self.df[idx]["boxes"][i]  # Use "boxes"
            labels.append(self.df[idx]["labels"][i])  # Use "labels"
            boxes.append(bbs)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image.double(), target, self.df[idx]["filename"]



class GbRoiDatasetNew(Dataset):
    """ GB region of interest dataset. """
    def __init__(self, img_dir, imgs, img_transforms=None):
        #with open(json_file, "r") as f:
        #    df = json.load(f)
        self.df = imgs
        self.img_dir = img_dir
        self.transforms = img_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.img_dir, self.df[idx])
        image = io.imread(img_name)
        #image = Image.open(img_name)
        #image = np.array(image)
        """
        num_objs = len(self.df[idx]["bbs"])
        labels = []
        boxes = []
        for i in range(num_objs):
            bbs = self.df[idx]["bbs"][i]
            if bbs[0] in ['abn', 'nml']:
                labels.append(LABEL_ENUM[bbs[0]])
                boxes.append(bbs[1])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        """
        target = {}
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image.double(), self.df[idx]

if __name__ == "__main__":
    VAL_IMG_DIR = "data_new/gb_imgs"
    VAL_JSON = "data_new/gb.json"
    with open(VAL_JSON, "r") as f:
        df = json.load(f)
    img_transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    dataset = GbRoiDataset(VAL_IMG_DIR, df, img_transforms = img_transforms)
    loader = DataLoader(dataset, batch_size=1, collate_fn=utils.collate_fn)
    images, targets = next(iter(loader))
    #print(images[0, 0, 150:160, 150:160])
    print(images[0][0,150:160, 150:160])
    targets = [{k: v for k, v in t.items()} for t in targets]
    print(targets)

