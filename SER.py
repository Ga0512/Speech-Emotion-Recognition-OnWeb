# pip install librosa==0.9.2 soundfile numpy scikit-learn pyaudio

import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
  with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))

  return result

import pickle

with open('model/mlp_model.pkl', 'rb') as m:
    model = pickle.load(m)


emoji_mapping = {
    'calm': 'Calm 😌',
    'happy': 'Happy 😃',
    'fearful': 'Fearful 😱',
    'disgust': 'Disgust 🤢',
    'neutral': 'Neutral 😐',
    'sad': 'Sad 😢',
    'angry': 'Angry 😡',
    'surprised': 'Surprised 😲'
}

def predict(file):
    features = extract_feature(file, mfcc=True, chroma=True, mel=True)
    
    result = emoji_mapping.get(model.predict([features])[0])
    return result

