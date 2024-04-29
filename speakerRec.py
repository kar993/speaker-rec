!pip install tensorflow_io
!pip install tensorflow_hub
import tensorflow as tf
import numpy as np
import tensorflow_hub as thub
import tensorflow_io as tfio
from google.colab import files
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split

vggish_model = thub.load("https://tfhub.dev/google/vggish/1")

def extracting_features(audio):
  audio=tf.cast(audio,tf.float32)
  audio /= 32768.0
  features = vggish_model(audio)
  print(features.shape)
  return features
def load_audio(file_path):
    audio = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio, desired_channels=1)
    audio = tf.squeeze(audio, axis=-1)
    return audio
karthik_audio_path="krt.wav"
srd_audio_path="srd.wav"

karthik_val_audio_path="krt-val.wav"
srd_val_audio_path="srd-val.wav"


karthik_audio=load_audio(karthik_audio_path)
srd_audio=load_audio(srd_audio_path)

karthik_val_audio=load_audio(karthik_val_audio_path)
srd_val_audio=load_audio(srd_val_audio_path)


kar_features=extracting_features(karthik_audio)
srd_features=extracting_features(srd_audio)

kar_val_features=extracting_features(karthik_val_audio)
srd_val_features=extracting_features(srd_val_audio)

features = tf.concat([srd_features, kar_features], axis=0)
labels = tf.concat([tf.zeros(srd_features.shape[0]), tf.ones(kar_features.shape[0])], axis=0)

val_features = tf.concat([srd_val_features, kar_val_features], axis=0)
val_labels = tf.concat([tf.zeros(srd_val_features.shape[0]), tf.ones(kar_val_features.shape[0])], axis=0)

print(features.shape)
print(labels.shape)
tf.reduce_max(features,0)

def expand(tensor,label):
  tensor=tf.expand_dims(tensor,0)
  return tensor,label

dataset=tf.data.Dataset.from_tensor_slices((features,labels)).map(expand).shuffle(buffer_size=100).batch(1)
val_dataset=tf.data.Dataset.from_tensor_slices((val_features,val_labels)).map(expand).shuffle(buffer_size=100).batch(1)
for data in dataset.take(1):
  print(data[0].shape)
for val_data in val_dataset.take(1):
  print(val_data[0].shape)


model = Sequential([
    Dense(128, activation='relu', name='dense_1'),
    tf.keras.layers.Conv1D(8,3,padding='same',input_shape=(1,128)),
    Dense(128, activation='relu', name='dense_2'),
    tf.keras.layers.Conv1D(16,3,padding='same'),
    tf.keras.layers.Flatten(),
    Dense(1, activation='sigmoid', name='output')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(dataset, epochs=1000,verbose=1)




test_loss, test_acc = model.evaluate(val_dataset)
print(f'Test accuracy: {test_acc}')