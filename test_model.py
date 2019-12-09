from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json

import numpy as np
from PIL import Image

json_file = open('test_uniform_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('test_uniform_model.h5')


data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
image = Image.open('./data-uniform/img_uniform02.jpg')
image = image.resize((150, 150))
image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

print(loaded_model.predict(data))