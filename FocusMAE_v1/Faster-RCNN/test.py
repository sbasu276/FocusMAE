from __future__ import print_function, division
import argparse
import os
import cv2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import transforms as T
import utils
from torch.optim.lr_scheduler import StepLR
from skimage import io, transform
from dataloader import GbRoiDatasetTest, GbClassDataset
from model import FasterRcnn
from sklearn.metrics import confusion_matrix, accuracy_score, \
        precision_score, recall_score, average_precision_score
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def parse():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--img_dir', dest="img_dir", default="path to image dir")
    parser.add_argument('--meta_file', dest="meta_file", default="path to meta file")
    parser.add_argument('--label_file', dest="label_file", default="path to label file")
    parser.add_argument('--height', dest="height", default=256, type=int)
    parser.add_argument('--width', dest="width", default=256, type=int)
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--comma_split', action='store_true')
    parser.add_argument('--out_dir', dest="out_dir", default="path to out dir")
    parser.add_argument('--batch_size', dest="batch_size", default=1, type=int)
    parser.add_argument('--save_dir', dest="save_dir", default="path to save dir")
    parser.add_argument('--save_name', dest="save_name", default="mod")
    parser.add_argument('--model_dir', dest="model_dir", default="models")
    parser.add_argument('--model_name', dest="model_name", default="mod")
    parser.add_argument('--pretrained_weights', dest="pretrained_weights", default="path to faster rcnn pretrained weights")
    args = parser.parse_args()
    return args


def get_iou(bb1, bb2):
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def save_stat(d, fname, args):
    res = d["results"]
    tp, fp, fn = 0, 0, 0
    ious = []
    df = {}
    tmp_data = []
    save_path = os.path.join(args.save_dir, str(args.save_name))
    #save_path_1 = os.path.join(args.save_dir, str(int(args.set_num)+1))
    os.makedirs(save_path, exist_ok=True)
    #os.makedirs(save_path_1, exist_ok=True)
    for e in res:
        pred = e["Boxes"]
        gb = [int(x) for x in e["Gold"][0]]
        image = cv2.imread("%s/%s"%(args.img_dir, e["image_id"]))
        if args.resize:
            image = cv2.resize(image, (args.width, args.height), interpolation = cv2.INTER_NEAREST)
        tmp = {}
        tmp["image_id"] = e["image_id"]
        tmp["Boxes"] = e["Boxes"].tolist()
        tmp["Labels"] = e["Labels"].tolist()
        #tmp["PredictionString"] = e["PredictionString"]
        tmp["Gold"] = e["Gold"][0]
        if pred.shape[0] > 0:
            mx_iou = 0
            mx_idx = 0
            for idx, pb in enumerate(pred):
                xp, yp = (pb[0]+pb[2])/2, (pb[1]+pb[3])/2
                if (gb[0] <= xp and xp <= gb[2]) and (gb[1] <= yp and yp <= gb[3]):
                    tp += 1
                else:
                    fp += 1
                if mx_iou < get_iou(pb, gb):
                    mx_iou = get_iou(pb, gb)
                    mx_idx = idx
                cv2.rectangle(image,(pb[0], pb[1]), \
                            (pb[2], pb[3]),(0,0,255),3)

            #img = image[int(0.95*pred[mx_idx][1]):int(1.05*pred[mx_idx][3]), \
            #                int(0.95*pred[mx_idx][0]):int(1.05*pred[mx_idx][2])]
            #cv2.imwrite("%s/%s"%(save_path, e["image_id"]), img)
            #cv2.rectangle(image,(pred[mx_idx][0], pred[mx_idx][1]), \
            #                (pred[mx_idx][2], pred[mx_idx][3]),(0,0,255),3)
            ious.append(get_iou(pb, gb))
            tmp["iou"] = mx_iou
            tmp["bb_idx"] = mx_idx
        else:
            fn += 1
            tmp["iou"] = 0.0
            cv2.imwrite("%s/%s"%(save_path, e["image_id"]), image)
            ious.append(0.0)
            
        tmp_data.append(tmp)
        cv2.rectangle(image,(gb[0], gb[1]),(gb[2], gb[3]),(0,255,0),3)
        # cv2.putText(image, "IOU = {:0.4}".format(float(tmp["iou"])), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (50,235,255), 5)
        # cv2.putText(image, "- Gold", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,235,0), 5)
        cv2.putText(image, "- Pred", (20,150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 5)
        cv2.imwrite("%s/%s"%(save_path, e["image_id"]), image)
    
    ious = np.array(ious)
    df["id"] = args.save_name
    df["miou"] = ious.mean()
    df["map"] = tp/(tp+fp)
    df["mar"] = tp/(tp+fn)
    df["results"] = tmp_data
    # print("%.4f, %.4f, %.4f"%(df["miou"], df["map"], df["mar"]))
    with open(fname, "w") as f:
        json.dump(df, f, indent=2)


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(\
                                j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


def main(args):
    
    if args.resize:
        img_transforms = T.Compose([T.Resize((args.width, args.height)), T.ToTensor()])
    else:
        img_transforms = T.Compose([T.ToTensor()])

    #val_set_file = os.path.join("data_new/cls_split", "val_%s.txt"%args.set_num)
    #os.path.join(args.set_dir, "val_%s.txt"%args.set_num)
    #val_set_file = "data_new/val_labels.txt"
    
    val_set_file = args.label_file 
    with open(args.meta_file, "r") as f:
        df = json.load(f)
    fnames = []
    labels = []
    y_true = {}
    with open(val_set_file, "r") as f:
        for e in f.readlines():
            # print('e: ',e)
            e = e.strip()
            labels.append(e)
            k, v = e.split(",")
            y_true[k] = int(v)
            args.comma_split=True
            if args.comma_split:
                e = e.split(",")[0]
            fnames.append(e)
    dset = {}
    for e in fnames:
        if e in df.keys():
            dset[e] = df[e]
    print('len of datasetset: ', len(dset))
    dataset = GbRoiDatasetTest(args.img_dir, dset, img_transforms=img_transforms)
    # print('dataset loader: ', len(dataset))
    #dataset = GbClassDataset(args.img_dir, dset, labels, img_transforms=img_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)

    net =  FasterRcnn(num_classes=2, train=False)
    # weight_file = "%s/mod_0.pth"%(args.model_dir)#, args.set_num)
    weight_file = args.pretrained_weights
    net.load_state_dict(torch.load(weight_file))
    net = net.float().cuda()

    detection_threshold = 0.25
    results = []
    # print('loader length: ', len(loader))
    
    for images, targets, image_ids in loader:
        # print(image_ids)
        images = [image.float().cuda() for image in images]
        targets = [{k: v.cuda() for k, v in t.items()} for t in targets]
        outputs = net(images)
        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            labels = outputs[i]['labels'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]
            if len(labels)==0:
                labels = np.array([3])
            # result = {
            #     'image_id': image_id,
            #     'Boxes': boxes,
            #     'Scores': scores,
            #     'Labels': labels,
            #     #'Img_Pred': max(labels)-1,
            #     #'PredictionString': format_prediction_string(boxes, scores),
            #     # 'Gold': targets[i]["boxes"].tolist()
            # }
            result = {
                'image_id': image_id,
                'Boxes': boxes.tolist(),  # Convert to list
                'Scores': scores.tolist(),  # Convert to list
                'Labels': labels.tolist()  # Convert to list
            }
            results.append(result)
    #true, pred = [], []
    #y_pred = {}
    #for res in results:
    #    fname = res["image_id"]
    #    true.append(y_true[fname])
    #    pred.append(res["Img_Pred"])
    #    y_pred[fname] = res["Img_Pred"]
    #print("Acc: %.4f"%accuracy_score(true, pred))
    #print(confusion_matrix(true, pred))
    
    #print(y_true)
    #print("---")
    #print(y_pred)


    d = {"results": results}
    # print('d: ',d)
    
    os.makedirs(args.out_dir, exist_ok=True)
    fname = os.path.join(args.out_dir, "res_frcnn_bbox.json")
    
    with open(fname, "w") as f:
        json.dump(d, f, indent=4)  # Save results in JSON format with indentation

    print(f"Results saved to {fname}")
    
    # save_stat(d, fname, args)

if __name__ == "__main__":
    args = parse()
    #print(args)
    main(args)
