import webbrowser
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

# Load the face detection and emotion classification models
face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier = load_model('./Emotion_Detection.h5')

# Emotion labels
class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion-to-YouTube music mapping
emotion_music = {
    'Angry': 'https://www.youtube.com/watch?v=8EJ3zbKTWQ8',  # Example: Energetic song
    'Happy': 'https://www.youtube.com/watch?v=A56mZxSXKn8',  # Example: Cheerful song
    'Neutral': 'https://www.youtube.com/watch?v=3JWTaaS7LdU',  # Example: Calm song
    'Sad': 'https://www.youtube.com/watch?v=09R8_2nJtjg',  # Example: Sad song
    'Surprise': 'https://www.youtube.com/watch?v=V2hlQkVJZhE'  # Example: Exciting song
}

cap = cv2.VideoCapture(0)
detected_emotion = "No Emotion Detected"

while True:
    ret, frame = cap.read()
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
            detected_emotion = class_labels[preds.argmax()]  # Store detected emotion
            label_position = (x, y)
            cv2.putText(frame, detected_emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Emotion Detector', frame)

    # Press 's' to stop and display detected emotion
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print(f"Detected Emotion: {detected_emotion}")  # Print detected emotion

        # Open corresponding song on YouTube
        if detected_emotion in emotion_music:
            webbrowser.open(emotion_music[detected_emotion])

        break

cap.release()
cv2.destroyAllWindows()

