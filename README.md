# Project - Emotion Detection & Music Recommendation

## Overview  
This project detects facial emotions using deep learning and maps suitable songs based on the detected emotion. While the model is functional, its accuracy can be improved with further fine-tuning and better dataset preprocessing.

## Dataset  
The model is trained using the **Facial Expression Recognition Challenge** dataset from Kaggle:  
[Facial Expression Recognition Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

## Implementation  

### Training the Model  
- Run `train.py` to train the model on facial emotion recognition.  
- The number of emotion classes can be modified.  
- Experimenting with different **pre-trained models** may improve accuracy.  

### Emotion Detection & Music Mapping  
- `test.py` detects emotions from facial expressions.  
- Based on the detected emotion, a relevant song is recommended.  
- The model may need improvements for better accuracy.  

## How to Run the Project  

## Clone the Repository  
```bash
git clone <repo-link>
cd music_emotion
```

## Train the model
```bash
python train.py
```
## Run Emotion Detection and Music Mapping
```bash
python test.py
```

## Future Improvements
- Enhancing model accuracy through better data augmentation and preprocessing.
- Trying out different deep learning architectures for emotion recognition.
- Expanding the song recommendation system with a larger music database.
- Implementing real-time emotion detection with live webcam input.

This project is a step towards integrating AI-driven emotion detection with personalized music recommendations, making interactions more engaging. 🚀🎶
