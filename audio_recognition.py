# -*- coding: utf-8 -*-
"""audio recognition.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bZdr5hizXpbk1fDYiqaxc2CCO8QwchrG
"""

#getting audio files from drive

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
os.chdir('/content/drive/MyDrive/audio')

#

os.listdir()

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


krtAudio, krtsr = librosa.load('krt.wav',sr=None)
srdAudio, srdsr = librosa.load('srd.wav',sr=None)
krtvalAudio, krtvalsr = librosa.load('krtval.wav',sr=None)
srdvalAudio, srdvalsr = librosa.load('srdval.wav',sr=None)

import os

# Function to split audio into smaller chunks
def split_audio(audio_data, sr, chunk_duration=1):
    chunk_samples = int(chunk_duration * sr)  # Duration of each chunk in samples
    return [audio_data[i:i+chunk_samples] for i in range(0, len(audio_data), chunk_samples)]

# Ensure the directories for each dataset exist within the 'melspectrograms' folder
os.makedirs('melspectrograms/krt_train', exist_ok=True)
os.makedirs('melspectrograms/krt_val', exist_ok=True)
os.makedirs('melspectrograms/srd_train', exist_ok=True)
os.makedirs('melspectrograms/srd_val', exist_ok=True)

# Function to generate and save Mel-spectrogram for each audio chunk
def save_melspectrogram_chunk(audio_chunk, sr, chunk_number, base_filename, category):
    # Generate Mel-spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram_dB = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Plot the Mel-spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8000)
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Mel-Spectrogram for {base_filename} - Chunk {chunk_number}')
    plt.tight_layout()

    # Define the folder to save based on the category
    output_folder = f'melspectrograms/{category}'
    output_filename = f'{output_folder}/{base_filename}_chunk_{chunk_number}.png'

    # Save the spectrogram as a PNG file
    plt.savefig(output_filename, format='png')
    plt.close()

# Example: Load your audio files
krt_train_audio, krt_train_sr = librosa.load('krt.wav', sr=None)
srd_train_audio, srd_train_sr = librosa.load('srd.wav', sr=None)
krt_val_audio, krt_val_sr = librosa.load('krtval.wav', sr=None)
srd_val_audio, srd_val_sr = librosa.load('srdval.wav', sr=None)

# Split audio into chunks (5-second chunks in this example)
krt_train_chunks = split_audio(krt_train_audio, krt_train_sr, chunk_duration=1)
srd_train_chunks = split_audio(srd_train_audio, srd_train_sr, chunk_duration=1)
krt_val_chunks = split_audio(krt_val_audio, krt_val_sr, chunk_duration=1)
srd_val_chunks = split_audio(srd_val_audio, srd_val_sr, chunk_duration=1)

# Save Mel-spectrograms for each chunk of KRT train audio
for i, chunk in enumerate(krt_train_chunks):
    save_melspectrogram_chunk(chunk, krt_train_sr, i+1, 'krt_train', 'krt_train')

# Save Mel-spectrograms for each chunk of SRD train audio
for i, chunk in enumerate(srd_train_chunks):
    save_melspectrogram_chunk(chunk, srd_train_sr, i+1, 'srd_train', 'srd_train')

# Save Mel-spectrograms for each chunk of KRT validation audio
for i, chunk in enumerate(krt_val_chunks):
    save_melspectrogram_chunk(chunk, krt_val_sr, i+1, 'krt_val', 'krt_val')

# Save Mel-spectrograms for each chunk of SRD validation audio
for i, chunk in enumerate(srd_val_chunks):
    save_melspectrogram_chunk(chunk, srd_val_sr, i+1, 'srd_val', 'srd_val')

krt_train_dir = '/content/drive/MyDrive/audio/melspectrograms/krt_train'
krt_val_dir = '/content/drive/MyDrive/audio/melspectrograms/krt_val'
srd_train_dir = '/content/drive/MyDrive/audio/melspectrograms/srd_train'
srd_val_dir = '/content/drive/MyDrive/audio/melspectrograms/srd_val'

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ImageDataGenerator with data augmentation for training
train_datagen = ImageDataGenerator(rescale=1./255,  # Normalize pixel values
                                   rotation_range=20,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

# ImageDataGenerator for validation (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load training data from both KRT and SRD training folders
train_generator = train_datagen.flow_from_directory(
    directory='/content/drive/MyDrive/audio/melspectrograms',
    classes=['krt_train', 'srd_train'],  # The folders in the main directory
    target_size=(128, 128),  # Resize all images to 128x128
    batch_size=8,
    class_mode='binary')  # For binary classification, 'categorical' for multiple classes

# Load validation data
val_generator = val_datagen.flow_from_directory(
    directory='/content/drive/MyDrive/audio/melspectrograms',
    classes=['krt_val', 'srd_val'],
    target_size=(128, 128),
    batch_size=8,
    class_mode='binary')

import tensorflow as tf
from tensorflow.keras import layers, models

# Define a simple CNN model
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Use binary cross-entropy for binary classification
              metrics=['accuracy'])

model.summary()

# Train the CNN
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,  # Number of training samples / batch size
    epochs=15,  # Adjust as needed
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size)

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")

# Plot training history
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()