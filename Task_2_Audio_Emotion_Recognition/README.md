# Audio Emotion Recognition Engine

Classifies human emotional states from raw speech audio using MFCC feature 
extraction and a Convolutional Neural Network (CNN).

## Dataset
- **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 1,440 audio files across 8 emotions: angry, calm, disgust, fearful, 
  happy, neutral, sad, surprised
- 24 professional actors (12M/12F), 48kHz WAV format
- Source: https://zenodo.org/records/1188976 (CC BY-NC-SA 4.0)

## Approach
1. Load audio at 22050Hz, fixed 3-second duration (pad/truncate)
2. Extract 40 MFCC coefficients per file → shape (40, 130)
3. Per-feature normalization across training set
4. CNN: Conv2D(32) → Conv2D(64) → Dense(128) → Softmax(8)
5. EarlyStopping + ReduceLROnPlateau callbacks

## Why accuracy is limited
RAVDESS contains only 1,440 samples across 8 classes. Neutral has 96 samples 
(half of other classes) due to no "strong intensity" variant — the model 
learned to never predict it. Published CNNs on this dataset typically achieve 
50–65%; higher accuracy requires transfer learning or larger combined datasets.

## How to run
Open `notebook/audio_emotion_recognition.ipynb` in Google Colab.
Download RAVDESS via Kaggle: `uwrfkaggler/ravdess-emotional-speech-audio`

**Live mic prediction output:**
```
🎤 Recording for 3 seconds... SPEAK NOW!
✅ Recording complete. Processing...

🎯 Predicted emotion: DISGUST
📊 Confidence: 59.1%

All probabilities:
  angry     : 0.022 ██
  calm      : 0.177 ███
  disgust   : 0.591 ████████████
  fearful   : 0.010
  happy     : 0.001
  neutral   : 0.001
  sad       : 0.198 ███
  surprised : 0.000
```

## Tech Stack
Python · TensorFlow/Keras · librosa · scikit-learn · NumPy · Matplotlib
