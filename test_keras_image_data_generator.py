from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

image_relevant_path='data/train/cats/cat.0.jpg'

if not os.path.exists(image_relevant_path):
    print('check image path : ' + image_relevant_path)
    exit('no image file')

img = load_img(image_relevant_path)  # PIL 이미지
x = img_to_array(img)  # (3, 150, 150) 크기의 NumPy 배열
x = x.reshape((1,) + x.shape)  # (1, 3, 150, 150) 크기의 NumPy 배열

# 아래 .flow() 함수는 임의 변환된 이미지를 배치 단위로 생성해서
# 지정된 `preview/` 폴더에 저장합니다.

directory_name = 'preview'

if not os.path.isdir(directory_name):
    os.makedirs(directory_name)

i = 0
for batch in datagen.flow(x, batch_size=1
                        , save_to_dir=directory_name
                        , save_prefix='cat'
                        , save_format='jpeg'):
    i += 1
    if i > 20:
        break  # 이미지 20장을 생성하고 마칩니다
