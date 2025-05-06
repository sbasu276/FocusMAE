
from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import transforms as T
import utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from dataloader import GbRoiDataset


class FasterRcnn(nn.Module):
    def __init__(self, num_classes=2, train=True):
        super(FasterRcnn, self).__init__()
        #self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model = fasterrcnn_resnet50_fpn(
                        pretrained=True, 
                        rpn_pre_nms_top_n_train=5, 
                        rpn_pre_nms_top_n_test=5, 
                        rpn_post_nms_top_n_train=5, 
                        rpn_post_nms_top_n_test=5
                    )
        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.is_training = train

    def forward(self, images, targets=None):
        if self.is_training:
            self.model.train()
            out = self.model(images, targets)
        else:
            self.model.eval()
            out = self.model(images)
        return out

if __name__ == "__main__":
    VAL_IMG_DIR = "gb_data/val_image"
    VAL_JSON = "gb_data/val.json"
    img_transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    dataset = GbRoiDataset(VAL_IMG_DIR, VAL_JSON, img_transforms = img_transforms)
    loader = DataLoader(dataset, batch_size=5, collate_fn=utils.collate_fn)
    images, targets = next(iter(loader))
    model = FasterRcnn() #fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.float()
    images = [image.float() for image in images]
    targets = [{k: v for k, v in t.items()} for t in targets]
    out = model(images, targets)
    print(out)


