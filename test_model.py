from tensorflow.keras.models import load_model

import numpy as np
from PIL import Image

model = load_model('uniform_model.h5')

data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)

image = Image.open('./test3.jpg')
image = image.resize((150, 150))
image_array = np.asarray(image)

normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

print(model.predict(data))