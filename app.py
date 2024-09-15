from flask import Flask, render_template, Response, jsonify
import cv2
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Load a pre-trained emotion recognition model
emotion_model = load_model('model.h5')
# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load music data
Music_Player = pd.read_csv("C:/Users/KUMARAN/OneDrive/Desktop/Spotify_Music_data_to_identify_the_moods/data_moods.csv")
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity']]

# Global variables to control video feed
video_active = True
last_emotion = ""

def detect_emotions(frame):
    global last_emotion  # Use the global variable to store the last detected emotion
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
        last_emotion = predicted_emotion  # Update the last emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame

def gen_frames():
    global video_active
    cap = cv2.VideoCapture(0)
    while video_active:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_emotions(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    global video_active, last_emotion
    video_active = True  # Reset video_active to True on page load
    last_emotion = ""    # Reset last_emotion on page load
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    global video_active
    video_active = False
    return jsonify({'emotion': last_emotion})

@app.route('/get_recommendations/<emotion>')
def get_recommendations(emotion):
    recommendations = recommend_songs(emotion)
    return {'recommendations': recommendations}

def recommend_songs(pred_class):
    if pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif pred_class in ['Happy', 'Sad']:
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif pred_class in ['Fear', 'Angry']:
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    else:  # Surprise or Neutral
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    
    Play = Play.sort_values(by="popularity", ascending=False).head(5).reset_index(drop=True)
    return Play.to_dict(orient='records')

if __name__ == "__main__":
    app.run(debug=True)