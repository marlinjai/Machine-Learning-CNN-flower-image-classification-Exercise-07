import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore

# Load the trained model
model_name = 'model_with_dropout_0_2'  # Change this to the model you want to use
model = load_model(f"{model_name}.keras")

# Load and preprocess the image
sunflower_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=sunflower_url)
img_height = 180
img_width = 180

img = image.load_img(sunflower_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Create a batch

# Make predictions
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

# Define class names (these should match the classes used during training)
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']  # Adjust this list as needed

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
)
