import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers  # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input  # type: ignore
import matplotlib.pyplot as plt
import os
import PIL
import pathlib

# Datenvorbereitung
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Datenaugmentation
data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Funktion zum Erstellen und Kompilieren des Modells
def create_model(with_dropout, dropout_rate=0.2):
    model_layers = [
      Input(shape=(img_height, img_width, 3)),  # Define input shape here
      data_augmentation,
      layers.Rescaling(1./255),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(5)
    ]

    if with_dropout:
        # Insert dropout layers right after MaxPooling2D layers
        model_layers.insert(5, layers.Dropout(dropout_rate))  # After first MaxPooling2D
        model_layers.insert(8, layers.Dropout(dropout_rate))  # After second MaxPooling2D
        model_layers.insert(11, layers.Dropout(dropout_rate))  # After third MaxPooling2D

    model = Sequential(model_layers)

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

# Training und Speichern der Modelle
def train_and_save_model(model_name, with_dropout, dropout_rate=0.2):
    model = create_model(with_dropout, dropout_rate)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15
    )
    # Save the model in the new Keras format
    model.save(f"{model_name}.keras")
    with open(f"{model_name}_history.npy", 'wb') as f:
        np.save(f, history.history)

# Modell ohne Dropout
train_and_save_model('model_without_dropout', with_dropout=False)

# Modell mit Dropout 0.2
train_and_save_model('model_with_dropout_0_2', with_dropout=True, dropout_rate=0.2)

# Modell mit Dropout 0.5
train_and_save_model('model_with_dropout_0_5', with_dropout=True, dropout_rate=0.5)
