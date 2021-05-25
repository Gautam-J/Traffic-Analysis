import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, Counter

import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.python.saved_model import tag_constants

import core.utils as utils
from core.config import cfg

from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

from flask import Flask, Response, render_template, request, redirect, url_for, send_file
import urllib.request

WEIGHTS = './checkpoints/yolov4-tiny-416'
# WEIGHTS = './checkpoints/yolov4-416'  # https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view
TINY = True if WEIGHTS.endswith('tiny-416') else False
IMG_SIZE = 416
CONF_THRESHOLD = 0.5  # object confidence threshold
IOU_THRESHOLD = 0.45  # IOU threshold for NMS
MAX_COSINE_DISTANCE = 0.4
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0
PEAK_TIME_WINDOW = 60  # seconds
upload_dir = 'temp'


def getAverageVehiclesPerSecond(arr, FPS):
    tots = []
    for i in range(0, len(arr) - FPS + 1, FPS):
        sum_ = 0
        for j in range(i, i + FPS):
            sum_ += arr[j]

        tots.append(sum_ / FPS)

    return round(np.mean(tots), 2)


def getStart(arr, k):
    res = 0
    start = 0

    for i in range(len(arr) - k):
        sum_ = 0
        for j in range(i, i + k):
            sum_ += arr[j]

        if sum_ > res:
            res = sum_
            start = i

    return start


if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

# configure GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", MAX_COSINE_DISTANCE, NN_BUDGET)
# initialize tracker
tracker = Tracker(metric)

# load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(TINY)

saved_model_loaded = tf.saved_model.load(WEIGHTS, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

# read in all class names from config
class_names = utils.read_class_names(cfg.YOLO.CLASSES)

allowed_classes = [
    'car',
    'motorbike',
    'bus',
    'truck',
]

# initialize color map
cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

forwardDensity = []
backwardDensity = []
stoppedVehiclesList = []
gateCounter = []
peakTimeVehicles = []

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'


@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        video = request.files['video']
        url = request.form.get('url')

        if url != '':
            urllib.request.urlretrieve(url, os.path.join(upload_dir, 'video.mp4'))
        else:
            video.save(os.path.join(upload_dir, 'video.mp4'))

        return redirect(url_for('detection'))

    return render_template('index.html')


@app.route('/detection')
def detection():
    return render_template('detection.html')


@app.route('/download')
def download():
    return send_file(os.path.join(upload_dir, 'out.avi'), as_attachment=True)


def stream():
    global count, movingVehicles, stoppedVehicles, currentForwardDensity, currentBackwardDensity, fps
    global peakTimeStart, averageForwardDensity, averageBackwardDensity, averageVehicles, averageStopTime
    peakTimeStart = '0'
    averageForwardDensity = '0'
    averageBackwardDensity = '0'
    averageVehicles = '0'
    averageStopTime = '0'
    count = '0'
    movingVehicles = '0'
    stoppedVehicles = '0'
    currentForwardDensity = '0'
    currentBackwardDensity = '0'
    fps = '0'

    videoCapture = cv2.VideoCapture("temp/video.mp4")
    width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = int(videoCapture.get(cv2.CAP_PROP_FPS))
    peakTimeFrames = PEAK_TIME_WINDOW * FPS
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(upload_dir, 'out.avi'), codec, FPS, (width, height))
    pts = [deque(maxlen=30) for _ in range(1000)]
    densityForward = deque(maxlen=FPS)
    densityBackward = deque(maxlen=FPS)

    while True:
        ret, frame = videoCapture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image_data = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for _, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=IOU_THRESHOLD,
                score_threshold=CONF_THRESHOLD
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
            names = np.array(names)
            count = len(names)

            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, NMS_MAX_OVERLAP, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            moving = []
            stopped = []
            forward = []
            backward = []
            counter = []

            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                bbox = track.to_tlbr()
                class_name = track.get_class()

                color = colors[int(track.track_id) % len(colors)]
                color = [i * 255 for i in color]

                # draw bounding boxes
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 1)
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)), (int(bbox[0]) + (len(class_name) + len(str(track.track_id))) * 17, int(bbox[1])), color, -1)
                cv2.putText(frame, class_name + "-" + str(track.track_id), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 1)

                # calculate centroid
                center = (int(((bbox[0]) + (bbox[2])) / 2), int(((bbox[1]) + (bbox[3])) / 2))
                pts[track.track_id].append(center)

                # draw trails
                for j in range(1, len(pts[track.track_id])):
                    if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                        continue

                    cv2.line(frame, (pts[track.track_id][j - 1]), (pts[track.track_id][j]), color, 2)

                centerY = int(((bbox[1]) + (bbox[3])) / 2)
                if centerY <= int(height * 0.7 + height / 20) and centerY >= int(height * 0.7 - height / 20):
                    counter.append(int(track.track_id))

                if len(pts[track.track_id]) > 5 and abs(pts[track.track_id][-5][1] - pts[track.track_id][-1][1]) < 1:
                    stopped.append(track.track_id)
                    stoppedVehiclesList.append(track.track_id)
                else:
                    moving.append(track.track_id)

                if (pts[track.track_id][0][1] < pts[track.track_id][-1][1]):
                    backward.append(track.track_id)
                else:
                    forward.append(track.track_id)

            fps = 1.0 / (time.time() - start_time)

            movingVehicles = len(set(moving))
            stoppedVehicles = len(set(stopped))
            forwardVehicles = len(set(forward))
            backwardVehicles = len(set(backward))

            densityForward.append(forwardVehicles)
            densityBackward.append(backwardVehicles)

            currentForwardDensity = np.mean(densityForward)
            currentBackwardDensity = np.mean(densityBackward)
            forwardDensity.append(currentForwardDensity)
            backwardDensity.append(currentBackwardDensity)

            gateCounter.append(len(set(counter)))
            peakTimeVehicles.append(count)

            # print(f'[INFO] vehicles in frame {count} | moving {movingVehicles} | stopped {stoppedVehicles} | forward density {currentForwardDensity:.2f} | backward density {currentBackwardDensity:.2f} | fps {fps:.2f}')

            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(result)
            frame = cv2.imencode('.jpg', result)[1].tostring()

            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')

        else:
            videoCapture.release()
            out.release()

            counter = Counter(stoppedVehiclesList)
            t = [counts for (_, counts) in counter.items()]

            peakTimeStart = round(getStart(peakTimeVehicles, peakTimeFrames) / FPS, 2)
            min, s = divmod(peakTimeStart, 60)
            peakTimeStart = f'{min} min {s} sec'
            averageForwardDensity = round(np.mean(forwardDensity), 2)
            averageBackwardDensity = round(np.mean(backwardDensity), 2)
            averageVehicles = getAverageVehiclesPerSecond(gateCounter, FPS)
            averageStopTime = round(np.mean(t) / FPS, 2)

            break


@app.route('/video_feed')
def video_feed():
    return app.response_class(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/count_feed')
def count_feed():

    def generate():
        yield str(count)

    return Response(generate(), mimetype='text/html')


@app.route('/movingVehicles_feed')
def movingVehicles_feed():

    def generate():
        yield str(movingVehicles)

    return Response(generate(), mimetype='text/html')


@app.route('/stoppedVehicles_feed')
def stoppedVehicles_feed():

    def generate():
        yield str(stoppedVehicles)

    return Response(generate(), mimetype='text/html')


@app.route('/currentForwardDensity_feed')
def currentForwardDensity_feed():

    def generate():
        yield str(round(float(currentForwardDensity), 2))

    return Response(generate(), mimetype='text/html')


@app.route('/currentBackwardDensity_feed')
def currentBackwardDensity_feed():

    def generate():
        yield str(round(float(currentBackwardDensity), 2))

    return Response(generate(), mimetype='text/html')


@app.route('/fps_feed')
def fps_feed():

    def generate():
        yield str(round(float(fps), 2))

    return Response(generate(), mimetype='text/html')


@app.route('/averageVehiclesMean_feed')
def averageVehiclesMean_feed():

    def generate():
        yield str(averageVehicles)

    return Response(generate(), mimetype='text/html')


@app.route('/averageStopTime_feed')
def averageStopTime_feed():

    def generate():
        yield str(averageStopTime)

    return Response(generate(), mimetype='text/html')


@app.route('/averageDensityForwardMean_feed')
def averageDensityForwardMean_feed():

    def generate():
        yield str(averageForwardDensity)

    return Response(generate(), mimetype='text/html')


@app.route('/averageDensityBackwardMean_feed')
def averageDensityBackwardMean_feed():

    def generate():
        yield str(averageBackwardDensity)

    return Response(generate(), mimetype='text/html')


@app.route('/peakTimeStart_feed')
def peakTimeStart_feed():

    def generate():
        yield str(peakTimeStart)

    return Response(generate(), mimetype='text/html')


if __name__ == '__main__':
    app.run()
