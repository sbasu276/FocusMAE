import random
import torch
from skimage import transform
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class Resize(object):
    def __init__(self, size):
        self.output_size = size

    def __call__(self, image, target=None):
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w), anti_aliasing=True)
        h_factor = new_h/h
        w_factor = new_w/w
        if target:
            bbox = target["boxes"]
            bbox[:, 0] = bbox[:, 0]*w_factor
            bbox[:, 1] = bbox[:, 1]*h_factor
            bbox[:, 2] = bbox[:, 2]*w_factor
            bbox[:, 3] = bbox[:, 3]*h_factor
            bbox = torch.as_tensor(bbox, dtype=torch.int16)
            target["boxes"] = bbox
            target["area"] = (bbox[:, 2] - bbox[:, 0])*(bbox[:, 3] - bbox[:, 1])
            retval = (img, target)
        else:
            retval = img
        return retval


class ToTensor(object):
    def __call__(self, image, target=None):
        image = F.to_tensor(image)
        return image, target
