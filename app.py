import cv2
import pandas as pd
import math
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load a pre-trained emotion recognition model (modified to handle input shape warning)
# Make sure the model definition uses `Input` explicitly in its architecture
emotion_model = load_model('model.h5')

# Define emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def detect_emotions(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    predicted_emotion = ""
    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to the size expected by the model
        roi_gray = roi_gray.astype('float32') / 255  # Normalize the image
        
        roi_gray = roi_gray.reshape(1, 48, 48, 1)
        # Predict the emotion
        predictions = emotion_model.predict(roi_gray)
        max_index = predictions[0].argmax()
        predicted_emotion = emotion_labels[max_index]
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    return frame, predicted_emotion

# Load music player data
Music_Player = pd.read_csv("data_moods.csv")
Music_Player = Music_Player[['name', 'artist', 'mood', 'popularity']]
Music_Player["mood"].value_counts()

# Function to recommend songs based on emotion
def Recommend_Songs(pred_class):
    if pred_class == 'Disgust':
        Play = Music_Player[Music_Player['mood'] == 'Sad']
    elif pred_class == 'Happy' or pred_class == 'Sad':
        Play = Music_Player[Music_Player['mood'] == 'Happy']
    elif pred_class == 'Fear' or pred_class == 'Angry':
        Play = Music_Player[Music_Player['mood'] == 'Calm']
    elif pred_class == 'Surprise' or pred_class == 'Neutral':
        Play = Music_Player[Music_Player['mood'] == 'Energetic']
    
    Play = Play.sort_values(by="popularity", ascending=False)
    Play = Play[:5].reset_index(drop=True)
    print(Play)

# Main function to capture video and detect emotions
def main():
    all_emot = []
    cap = cv2.VideoCapture(0)  # Capture video from webcam
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, emot = detect_emotions(frame)
        all_emot.append(emot)
        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    # Find the most detected emotion
    max_count = max(all_emot.count(emotion) for emotion in emotion_labels)
    detected_emotion = max(all_emot, key=all_emot.count)
    
    print("\n\nYou look : ", detected_emotion)
    print("\n\nYou Listen to the Following Famous Songs to Lighten up Your Mood\n")
    Recommend_Songs(detected_emotion)

if __name__ == "__main__":
    main()
