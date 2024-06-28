import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Input # type: ignore
import matplotlib.pyplot as plt
import os
import PIL
import pathlib
from itertools import product

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
    layers.Input(shape=(img_height, img_width, 3)),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

def visualize_data_augmentation():
  plt.figure(figsize=(10, 10))
  for images, _ in train_ds.take(1):
    augmented_images = data_augmentation(images)
    for i in range(9):
      ax = plt.subplot(3, 3, i + 1)
      plt.imshow(augmented_images[i].numpy().astype("uint8"))
      plt.axis("off")
  plt.savefig('data_augmentation_examples.png')
  plt.show()

def create_model(with_dropout, dropout_rate, learning_rate, activation_function, with_data_augmentation):
  model_layers = [
    layers.Input(shape=(img_height, img_width, 3)),
    data_augmentation if with_data_augmentation else layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation=activation_function),
    layers.MaxPooling2D(),
    layers.Dropout(dropout_rate) if with_dropout else None,
    layers.Conv2D(32, 3, padding='same', activation=activation_function),
    layers.MaxPooling2D(),
    layers.Dropout(dropout_rate) if with_dropout else None,
    layers.Conv2D(64, 3, padding='same', activation=activation_function),
    layers.MaxPooling2D(),
    layers.Dropout(dropout_rate) if with_dropout else None,
    layers.Flatten(),
    layers.Dense(128, activation=activation_function),
    layers.Dense(5)
  ]

  # Remove None values from the list
  model_layers = [layer for layer in model_layers if layer is not None]

  model = Sequential(model_layers)

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
  return model

def visualize_and_save_training_results(history, model_name, learning_rate, activation_function, with_data_augmentation, with_dropout, dropout_rate, epochs):
  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']
  epochs_range = range(epochs)

  plt.figure(figsize=(16, 8))
  plt.subplot(1, 2, 1)
  plt.plot(epochs_range, acc, label='Training Accuracy')
  plt.plot(epochs_range, val_acc, label='Validation Accuracy')
  plt.legend(loc='lower right')
  plt.title(f'Training and Validation Accuracy\nLR: {learning_rate}, AF: {activation_function}, DA: {with_data_augmentation}, Dropout: {with_dropout} ({dropout_rate}), Epochs: {epochs}')

  plt.subplot(1, 2, 2)
  plt.plot(epochs_range, loss, label='Training Loss')
  plt.plot(epochs_range, val_loss, label='Validation Loss')
  plt.legend(loc='upper right')
  plt.title(f'Training and Validation Loss\nLR: {learning_rate}, AF: {activation_function}, DA: {with_data_augmentation}, Dropout: {with_dropout} ({dropout_rate}), Epochs: {epochs}')
  
  folder_path = f"./models/{model_name}"
  os.makedirs(folder_path, exist_ok=True)
  plt.savefig(f"{folder_path}/{model_name}_training_curves.png")
  plt.show()

def train_and_save_model(with_dropout, dropout_rate, learning_rate, activation_function, with_data_augmentation, epochs):
  # Generate model name based on parameters
  model_name = f"dropout{with_dropout}_rate{dropout_rate}_lr{learning_rate}_act{activation_function}_aug{with_data_augmentation}_epochs{epochs}"
  model = create_model(with_dropout, dropout_rate, learning_rate, activation_function, with_data_augmentation)
  
  callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'),
    tf.keras.callbacks.ModelCheckpoint(f"./models/{model_name}/{model_name}_best.keras", save_best_only=True, monitor='val_loss')
  ]

  history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
  )
  
  folder_path = f"./models/{model_name}"
  os.makedirs(folder_path, exist_ok=True)
  model.save(f"{folder_path}/{model_name}.keras")
  with open(f"{folder_path}/{model_name}_history.npy", 'wb') as f:
    np.save(f, history.history)
  
  visualize_and_save_training_results(history, model_name, learning_rate, activation_function, with_data_augmentation, with_dropout, dropout_rate, epochs)
  if with_data_augmentation:
    visualize_data_augmentation()

# Define parameter grid
parameter_grid = {
  'with_dropout': [True, False],
  'dropout_rate': [0.2, 0.5],
  'learning_rate': [0.001, 0.0001],
  'activation_function': ['relu', 'sigmoid', 'tanh'],
  'with_data_augmentation': [True, False],
  'epochs': [10, 20]
}

# Generate all combinations of parameters
def generate_parameter_combinations(parameter_grid):
  keys, values = zip(*parameter_grid.items())
  return [dict(zip(keys, v)) for v in product(*values)]

parameter_combinations = generate_parameter_combinations(parameter_grid)

# Loop over all parameter combinations and train models
for params in parameter_combinations:
  train_and_save_model(**params)
