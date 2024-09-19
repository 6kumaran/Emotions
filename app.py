import streamlit as st
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load a pre-trained emotion recognition model
emotion_model = load_model('model.h5')  # Update with your model path

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load music data
Music_Player = pd.read_csv('data_moods.csv')  # Update with your CSV path
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
        
        # Predict the emotion
        predictions = emotion_model.predict(roi_gray)
        max_index = predictions[0].argmax()
        predicted_emotion = emotion_labels[max_index]
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame, predicted_emotion

def recommend_songs(pred_class):
    if pred_class == 'Disgust':
        mood = 'Sad'
    elif pred_class in ['Happy', 'Sad']:
        mood = 'Happy'
    elif pred_class in ['Fear', 'Angry']:
        mood = 'Calm'
    elif pred_class in ['Surprise', 'Neutral']:
        mood = 'Energetic'
    else:
        mood = 'Calm'

    play = Music_Player[Music_Player['mood'] == mood]
    play = play.sort_values(by="popularity", ascending=False).head(5).reset_index(drop=True)
    return play

# Streamlit app
st.title("Emotion Detection and Music Recommendation")

if st.button("Start Camera"):
    cap = cv2.VideoCapture(0)  # Capture video from webcam
    stframe = st.empty()
    
    all_emotions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Error: Could not read frame.")
            break
        
        frame, emot = detect_emotions(frame)
        all_emotions.append(emot)
        
        # Display frame in Streamlit
        stframe.image(frame, channels='BGR', use_column_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

    # Most frequent emotion
    if all_emotions:
        final_emotion = max(set(all_emotions), key=all_emotions.count)
        st.write(f"You look: {final_emotion}")
        st.write("Recommended songs to lighten up your mood:")
        recommendations = recommend_songs(final_emotion)
        st.write(recommendations)

    cv2.destroyAllWindows()
