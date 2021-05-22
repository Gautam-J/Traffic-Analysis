import cv2
import math
import torch
import random
import numpy as np

from utils.datasets import letterbox
from utils.utils import non_max_suppression, scale_coords, plot_one_box


def getNewImageSize(img_size, stride=32):
    # returns the input image size that is valid for yolov5 arch

    # img_size should be evenly divisible by max_stride of 32
    newSize = math.ceil(img_size / stride) * stride

    if newSize != img_size:
        print('[DEBUG] New image size', newSize)

    return newSize


IMG_SIZE = 640
IMG_SIZE = getNewImageSize(640)

WEIGHTS = 'weights/yolov5s.pt'
DEVICE = torch.device('cpu')

CONF_THRESHOLD = 0.4  # object confidence threshold
IOU_THRESHOLD = 0.5  # IOU threshold for NMS

model = torch.load(WEIGHTS, map_location=DEVICE)['model'].float()
model.to(DEVICE).eval()

classes = model.names if hasattr(model, 'names') else model.modules.names

# random colours for each class
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

videoCapture = cv2.VideoCapture('data/RoadTrafficVideo.mp4')

while True:
    _, img0 = videoCapture.read()

    img = letterbox(img0, new_shape=IMG_SIZE)[0]  # padded resize
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(DEVICE).float()
    img /= 255.0
    img = img.unsqueeze(0)

    pred = model(img, augment=False)[0]

    # apply Non Max Suppression
    pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD, fast=True)

    for i, det in enumerate(pred):
        s = ''
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %s, ' % (n, classes[int(c)])

            for *xyxy, conf, cls in det:
                label = '%s %.2f' % (classes[int(cls)], conf)
                plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)

    print(s)

    cv2.imshow('yolov5', img0)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
