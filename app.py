from flask import Flask, render_template, request, jsonify
import base64
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load a pre-trained emotion recognition model
emotion_model = load_model('model.h5')
# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load music data
Music_Player = pd.read_csv("data_moods.csv")  # Ensure this file is in the same directory as app.py

def detect_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    predicted_emotion = ""
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float32') / 255
        roi_gray = roi_gray.reshape(1, 48, 48, 1)
        predictions = emotion_model.predict(roi_gray)
        max_index = predictions[0].argmax()
        predicted_emotion = emotion_labels[max_index]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame, predicted_emotion

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data['image']

    # Decode the image
    header, encoded = image_data.split(',', 1)
    image = base64.b64decode(encoded)
    np_image = np.frombuffer(image, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Process the frame to detect emotions
    frame, emot = detect_emotions(frame)

    # Return the detected emotion
    return jsonify({"emotion": emot})

if __name__ == "__main__":
    app.run(debug=True)
