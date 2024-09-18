from flask import Flask, request, jsonify
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the model and data
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
Music_Player = pd.read_csv('data_moods.csv')
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity']]

# Define emotion detection and song recommendation functions
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

def Recommend_Songs(pred_class):
    if pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif pred_class in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif pred_class in ['Fear', 'Angry']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    else:
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    Play = Play.sort_values(by="popularity", ascending=False).head(5).reset_index(drop=True)
    return Play

# Define a route
@app.route('/')
def index():
    return 'Emotion Detection and Music Recommendation API'

# Run the application
if __name__ == "__main__":
    app.run()
