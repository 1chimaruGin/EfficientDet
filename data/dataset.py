""" COCO dataset (quick and dirty)

Hacked together by Ross Wightman
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data

import os
import cv2
import random
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

class CocoDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        ann_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``

    """

    def __init__(self, root, ann_file, transform=None):
        super(CocoDetection, self).__init__()
        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = root
        self.transform = transform
        self.yxyx = True   # expected for TF model, most PT are xyxy
        self.include_masks = False
        self.include_bboxes_ignore = False
        self.has_annotations = 'image_info' not in ann_file
        self.coco = None
        self.cat_ids = []
        self.cat_to_label = dict()
        self.img_ids = []
        self.img_ids_invalid = []
        self.img_infos = []
        self._load_annotations(ann_file)

    def _load_annotations(self, ann_file):
        assert self.coco is None
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        img_ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for img_id in sorted(self.coco.imgs.keys()):
            info = self.coco.loadImgs([img_id])[0]
            valid_annotation = not self.has_annotations or img_id in img_ids_with_ann
            if valid_annotation and min(info['width'], info['height']) >= 32:
                self.img_ids.append(img_id)
                self.img_infos.append(info)
            else:
                self.img_ids_invalid.append(img_id)

    def _parse_img_ann(self, img_id, img_info):
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        bboxes = []
        bboxes_ignore = []
        cls = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if self.include_masks and ann['area'] <= 0:
                continue
            if w < 1 or h < 1:
                continue

            # To subtract 1 or not, TF doesn't appear to do this so will keep it out for now.
            if self.yxyx:
                #bbox = [y1, x1, y1 + h - 1, x1 + w - 1]
                bbox = [y1, x1, y1 + h, x1 + w]
            else:
                #bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
                bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                if self.include_bboxes_ignore:
                    bboxes_ignore.append(bbox)
            else:
                bboxes.append(bbox)
                cls.append(self.cat_to_label[ann['category_id']] if self.cat_to_label else ann['category_id'])

        if bboxes:
            bboxes = np.array(bboxes, dtype=np.float32)
            cls = np.array(cls, dtype=np.int64)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
            cls = np.array([], dtype=np.int64)

        if self.include_bboxes_ignore:
            if bboxes_ignore:
                bboxes_ignore = np.array(bboxes_ignore, dtype=np.float32)
            else:
                bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(img_id=img_id, bbox=bboxes, cls=cls, img_size=(img_info['width'], img_info['height']))

        if self.include_bboxes_ignore:
            ann['bbox_ignore'] = bboxes_ignore

        return ann

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, annotations (target)).
        """
        img_id = self.img_ids[index]
        img_info = self.img_infos[index]
        if self.has_annotations:
            ann = self._parse_img_ann(img_id, img_info)
        else:
            ann = dict(img_id=img_id, img_size=(img_info['width'], img_info['height']))

        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img, ann = self.transform(img, ann)

        return img, ann

    def __len__(self):
        return len(self.img_ids)


class Custom_Dataset(data.Dataset):
    def __init__(self, root, data, image_ids, transform=None, test=False):
        self.root = root
        self.data = data 
        self.image_ids = image_ids
        self.transform = transform
        self.test = test

    def _load_data(self, index):
        image_id = self.image_ids[index]
        image = cv2.imread(f'{self.root}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0
        record = self.data[self.data['image_id'] == image_id]
        boxes = record[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        return image, boxes

    def _load_cutmix_data(self, index, imgsize=1024):
        w, h = imgsize, imgsize
        s = imgsize // 2

        xc, yc = [int(random.uniform(imgsize * .25, imgsize * .75)) for _ in range(2)]
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imgsize, imgsize, 3), 1, dtype=np.float32)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self._load_data(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b    

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)
        
        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:, 2] - result_boxes[:, 0]) * (result_boxes[:, 3] - result_boxes[:, 1]) > 0)]
        return result_image, result_boxes

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        if self.test or random.random() > 0.35:
            image, boxes = self._load_data(index)
        elif random.random() > 0.5:
            image, boxes = self._load_cutmix_data(index)
        else:
            image, boxes = self._load_cutmix_data(index)

        labels = torch.ones((boxes.shape[0]), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor(index)

        if self.transform:
            for i in range(10):
                sample = self.transform(**{
                    'image': image,
                    'bboxes': target['boxes'],
                    'labels': labels
                })
                if len(sample['bboxes']) > 0:
                    image = sample['image']
                    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
                    target['boxes'][:, [0, 1, 2, 3]] = target['boxes'][:, [1, 0, 3, 2]]
                    break
        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]