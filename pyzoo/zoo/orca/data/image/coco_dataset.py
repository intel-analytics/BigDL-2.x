#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import os
import json
import logging
import numpy as np
from PIL import Image
from os import path as osp
from pycocotools.coco import COCO


class COCODetection:
    """
    data_root: str, example "data_root='data/coco/'"
    splits_name: str
    classes: list['str']
    difficult: bool, False ignore coco json difficult value.
    """

    def __init__(self, data_root, ann_file, classes):
        self.CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
                        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
                        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.data_root = data_root
        self.ann_file = ann_file
        if classes:
            self.classes = classes
        else:
            self.classes = self.CLASSES
        self.data_infos = self.load_annotations(self.ann_file)
        self._ann_info = [self._load_label(idx) for idx in range(len(self))]
        self._img_path = [osp.join(self.data_root, img_info['filename'])
                          for img_info in self.data_infos]

    def __len__(self):
        return len(self.data_infos)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds(catNms=self.classes)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds()

        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def __iter__(self):
        return zip(self._img_path, self._ann_info)

    def __getitem__(self, index):
        img_path = self._img_path[index]
        img = self._read_image(img_path)
        return img, self._ann_info[index]

    def _load_label(self, index):
        gt_bboxes = []
        img_info = self.data_infos[index]
        img_id = img_info['id']
        ann_id = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_id)
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            else:
                label = self.cat2label[ann['category_id']]
            bbox = [int(x1), int(y1), int(x1 + w), int(y1 + h), int(label)]
            if ann.get('iscrowd', False):
                continue
            gt_bboxes.append(bbox)
        gt_bboxes = np.array(gt_bboxes).astype(np.uint8)
        return gt_bboxes

    def _read_image(self, image_path):
        try:
            img = Image.open(image_path)
            img = np.array(img)
            img = img.astype(np.uint8)
            return img
        except FileNotFoundError as e:
            return e


if __name__ == "__main__":
    data_dir = "/home/gtf/Datasets/VOC2007/JPEGImages"
    ann_info = '/home/gtf/Datasets/VOC2007/VOC2007.json'
    coco_data = COCODetection(data_dir, ann_info, classes=None)
    img, label = coco_data[0]
    print('label', label)
    print('data_len', len(coco_data))
