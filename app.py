from flask import Flask, render_template, Response
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

# Initialize the Flask app
app = Flask(__name__)

# Load the face detector and emotion model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the music data
Music_Player = pd.read_csv('data_moods.csv')
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity']]

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
    return predicted_emotion

def Recommend_Songs(emot):
    if emot == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif emot in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif emot in ['Fear', 'Angry']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    else:
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    Play = Play.sort_values(by="popularity", ascending=False).head(5)
    return Play[['name', 'artist']].to_dict(orient='records')

@app.route('/')
def index():
    return render_template('index.html')

def generate_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        predicted_emotion = detect_emotions(frame)
        # Process the frame for display if needed
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
