import cv2
import torch
import numpy as np

from utils.datasets import letterbox
from utils.utils import (
    non_max_suppression,
    scale_coords,
    plot_one_box,
    check_img_size
)

IMG_SIZE = 640
CONF_THRESHOLD = 0.4  # object confidence threshold
IOU_THRESHOLD = 0.5  # IOU threshold for NMS
OUT_FPS = 20.0
VIDEO_PATH = 'data/cut.mp4'
WEIGHTS = 'weights/yolov5s.pt'
DEVICE = torch.device('cpu')

IMG_SIZE = check_img_size(IMG_SIZE)
model = torch.load(WEIGHTS, map_location=DEVICE)['model'].float()
model.to(DEVICE).eval()

classes = model.names if hasattr(model, 'names') else model.modules.names
vehicleClasses = [
    'car',
    'motorcycle',
    'bus',
    'truck',
]
vehicleClassIndex = [classes.index(i) for i in vehicleClasses]

videoCapture = cv2.VideoCapture(VIDEO_PATH)
frameHeight, frameWidth = int(videoCapture.get(3)), int(videoCapture.get(4))
size = (frameHeight, frameWidth)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data/output.avi', fourcc, OUT_FPS, size)

while True:
    ret, img0 = videoCapture.read()

    if ret:
        img = letterbox(img0, new_shape=IMG_SIZE)[0]  # padded resize
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(DEVICE).float()
        img /= 255.0
        img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD,
                                   fast=True, classes=vehicleClassIndex)

        for i, det in enumerate(pred):
            s = ''

            # normalization gain whwh
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]

            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:],
                                          det[:, :4],
                                          img0.shape).round()

                nVehicles = 0
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %s, ' % (n, classes[int(c)])
                    nVehicles += n

                for *xyxy, conf, cls in det:
                    label = '%s %.2f' % (classes[int(cls)], conf)
                    plot_one_box(xyxy, img0, label=label, color=[255, 155, 0],
                                 line_thickness=1)

                s += 'Total Vehicles: %g' % nVehicles

        print(s)
        out.write(img0)

        cv2.imshow('yolov5', img0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

videoCapture.release()
out.release()
cv2.destroyAllWindows()

print('Saved Processed Video')
