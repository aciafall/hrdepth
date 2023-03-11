import os
from torch.utils.data import Dataset
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import cv2
import json


class Resize(object):
    def __init__(self, width=640, height=352):
        self.width = width
        self.height = height

    def __call__(self, image, boxes):
        # print(image,boxes)
        oral_w, oral_h = image.shape[1], image.shape[0]
        image = cv2.resize(image, (self.width, self.height))
        boxes[:, 0] = boxes[:, 0] / oral_w * self.width
        boxes[:, 1] = boxes[:, 1] / oral_h * self.height
        boxes[:, 2] = boxes[:, 2] / oral_w * self.width
        boxes[:, 3] = boxes[:, 3] / oral_h * self.height

        return image.transpose(2, 0, 1), boxes


class NuSceneDataset(Dataset):
    def __init__(self, data_dir, split, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.ids = split
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image_id = self.ids[index]
        boxes = self._get_annotation(image_id)
        image = self._read_image(image_id)
        if self.transform:
            image, boxes = self.transform(image, boxes)
        if self.target_transform:
            boxes = self.target_transform(boxes)

        return image, boxes, index

    def __len__(self):
        return len(self.ids)


    def _get_annotation(self, image_id):
        annotation_file = os.path.join(self.data_dir, "Annotations/samples", "%s.json" % image_id)
        with open(annotation_file, 'r') as fp:
            json_data = json.load(fp)
        shapes = json_data['shapes']
        boxes = []
        for shape in shapes:
            # VOC dataset format follows Matlab, in which indexes start from 0
            x_min, y_min = shape['points'][0]
            x_max, y_max = shape['points'][1]
            x1 = float(x_min) - 1
            y1 = float(y_min) - 1
            x2 = float(x_max) - 1
            y2 = float(y_max) - 1
            boxes.append([x1, y1, x2, y2, float(shape['label'])])

        boxes = sorted(boxes, key=lambda k: k[-1])

        return np.array(boxes, dtype=np.float32)

    def get_img_info(self, index):
        # img_id = self.ids[index]
        # annotation_file = os.path.join(self.data_dir, "Annotations/samples", "%s.json" % img_id)
        # with open(annotation_file, 'r') as fp:
        #     json_data = json.load(annotation_file)
        # anno = ET.parse(annotation_file).getroot()
        # size = anno.find("size")
        # im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))
        # return {"height": im_info[0], "width": im_info[1]}
        return {"height": 900, "width": 1600}

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "Images/samples", "%s.jpg" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image, dtype=np.float32)
        return image
