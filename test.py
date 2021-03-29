import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np

model = load_model("image_prediction.h5")
class_names = ['Papatya', 'Karahindiba', 'Güller', 'Ay çiçeği', 'Lale']
flower_url = input("Bir resim URL'si giriniz: ")
flower_name = input("Resmin ismini giriniz: ")
flower_path = tf.keras.utils.get_file(flower_name, origin=flower_url)

img_height = 180
img_width = 180

img = keras.preprocessing.image.load_img(
    flower_path, target_size=(img_height, img_width)
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0) # Create a batch

predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

print(
    "{:.2f} oranında bu resim bir {} resmi"
    .format(100 * np.max(score), class_names[np.argmax(score)])
)

