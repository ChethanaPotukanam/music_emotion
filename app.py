import streamlit as st
import cv2
import numpy as np
import webbrowser
import random
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Load models once
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')

# Emotion labels
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion-to-YouTube music mapping (Multiple songs for each emotion)
emotion_music = {
    'Angry': [
        'https://www.youtube.com/watch?v=ixlXwdmZ3Kg',
        'https://www.youtube.com/watch?v=coOXVfbP-JA',
        'https://www.youtube.com/watch?v=jGflUbPQfW8'
    ],
    'Happy': [
        'https://www.youtube.com/watch?v=cGc_NfiTxng',
        'https://www.youtube.com/watch?v=ZbZSe6N_BXs',
        'https://www.youtube.com/watch?v=ru0K8uYEZWw'
    ],
    'Neutral': [
        'https://www.youtube.com/watch?v=oAVhUAaVCVQ',
        'https://www.youtube.com/watch?v=OpQFFLBMEPI',
        'https://www.youtube.com/watch?v=1k8craCGpgs'
    ],
    'Sad': [
        'https://www.youtube.com/watch?v=YR12Z8f1Dh8',
        'https://www.youtube.com/watch?v=0yW7w8F2TVA',
        'https://www.youtube.com/watch?v=J_ub7Etch2U'
    ],
    'Surprise': [
        'https://www.youtube.com/watch?v=WXC4ScQ0YVg',
        'https://www.youtube.com/watch?v=3tmd-ClpJxA',
        'https://www.youtube.com/watch?v=L_jWHffIx5E'
    ]
}

# Streamlit UI
st.title("🎭 Emotion Detection & Music Recommendation 🎶")
st.write("Detect your emotion and play a song based on it!")

# Initialize session state variables
if "run_detection" not in st.session_state:
    st.session_state.run_detection = False
if "detected_emotion" not in st.session_state:
    st.session_state.detected_emotion = "No Emotion Detected"

# Buttons to start and stop detection
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Detection 🎥"):
        st.session_state.run_detection = True
with col2:
    if st.button("Stop & Show Emotion 🎭"):
        st.session_state.run_detection = False

frame_placeholder = st.empty()  # Placeholder for video stream

# Webcam processing
if st.session_state.run_detection:
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and st.session_state.run_detection:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video. Please check your webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                preds = classifier.predict(roi)[0]
                st.session_state.detected_emotion = class_labels[preds.argmax()]

        frame_placeholder.image(frame, channels="BGR", use_column_width=True)

    cap.release()
    cv2.destroyAllWindows()

# Show emotion and play music after stopping
if not st.session_state.run_detection and st.session_state.detected_emotion != "No Emotion Detected":
    st.write(f"**Detected Emotion: {st.session_state.detected_emotion}**")

    if st.session_state.detected_emotion in emotion_music:
        song_link = random.choice(emotion_music[st.session_state.detected_emotion])
        webbrowser.open(song_link)
        st.video(song_link)  # Show video in Streamlit

