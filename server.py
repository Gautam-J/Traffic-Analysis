import cv2
import time
from flask import Flask, Response, render_template

cam = cv2.VideoCapture(0)
app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')


def stream():
    global t
    while True:
        ret, frame = cam.read()
        if ret:
            frame = cv2.imencode('.jpg', frame)[1].tostring()
            t = str(time.time())
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')
        else:
            break


@app.route('/video_feed')
def video_feed():
    return app.response_class(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/time_feed')
def time_feed():

    def generate():
        yield t

    return Response(generate(), mimetype='text/html')


if __name__ == '__main__':
    app.run()
