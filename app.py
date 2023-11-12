from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from numpy import expand_dims
from keras_facenet import FaceNet
from imutils.video import VideoStream
import time
import joblib
import threading

app = Flask(__name__)

# Global variables for recognition control
recognize_status = {'status': 'stopped'}
recognize_lock = threading.Lock()

# Global variable to store the video processing thread
video_thread = None

# Load the FaceNet model
MyFaceNet = FaceNet()

# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the SVM model and label encoder
model = joblib.load('svm_model_7expressions.pkl')
out_encoder = joblib.load('label_encoder_7expressions.pkl')

def start_recognition():
    global recognize_status, video_thread
    with recognize_lock:
        recognize_status['status'] = 'started'

def stop_recognition():
    global recognize_status, video_thread
    with recognize_lock:
        recognize_status['status'] = 'stopped'

def generate_frames():
    cap = cv2.VideoCapture(0)  # Initialize the webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160))

        with recognize_lock:
            if recognize_status['status'] == 'started':
                for (x, y, w, h) in faces:
                    face = frame[y:y + h, x:x + w]
                    face = cv2.resize(face, (160, 160))
                    face = expand_dims(face, axis=0)

                    signature = MyFaceNet.embeddings(face)

                    yhat_encoded = model.predict(signature)
                    yhat_prob = model.predict_proba(signature)
                    class_index = yhat_encoded[0]
                    class_probability = yhat_prob[0, class_index] * 100
                    predict_name = out_encoder.inverse_transform(yhat_encoded)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    text = f'{predict_name[0]} ({class_probability:.2f}%)'
                    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/start_recognize', methods=['GET'])
def start_recognize():
    start_recognition()
    return jsonify(recognize_status)

@app.route('/stop_recognize', methods=['GET'])
def stop_recognize():
    stop_recognition()
    return jsonify(recognize_status)

if __name__ == '__main__':
    app.run(debug=True)
