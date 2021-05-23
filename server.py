import cv2
from flask import Flask, Response, render_template

cam = cv2.VideoCapture(0)
app = Flask(__name__)


def stream():
    while True:
        ret, frame = cam.read()
        if ret:
            frame = cv2.imencode('.jpg', frame)[1].tostring()
            yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + frame + b'\r\n')
        else:
            break


@app.route('/video_feed')
def video_feed():
    return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
