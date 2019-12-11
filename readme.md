# 목표

작은 데이터셋으로 강력한 이미지 분류 모델 설계하기

# 데이터 셋

https://www.kaggle.com/c/dogs-vs-cats/data

```
data
    +train
        +cats
            -cat.0.jpg
            -cat.1.jpg
            -cat.2.jpg
            ...
        +dogs
            -dog.0.jpg
            -dog.1.jpg
            -dog.2.jpg
            ...


    +validation
        +cats
            -cat.0.jpg
            -cat.1.jpg
            -cat.2.jpg
            ...
        +dogs
            -dog.0.jpg
            -dog.1.jpg
            -dog.2.jpg
            ...
```

- train 폴더에 검증 데이터와 겹치지않는 데이터 2000개 씩(개:2000, 고양이:2000)
- validation 폴더에 트레이닝 데이터와 겹치지 않는 데이터 400씩(개:400, 고양이:400)

# require
- pip install tensorflow # tensorflow 2.0 내장 keras 사용
- pip install image      # PIL
- pip install SciPy      # image transformations require


# 실행
1.
```
python test_keras_image_data_generator.py
keras ImageDataGenerator 테스트
```
2.
```
python classifier_from_little_data.py
first_try.h5 학습
```
3.
```
python classifier_from_little_data_using_pretraining_model
병목 특징을 이용한 bottleneck_features_train.npy, bottleneck_features_validation.npy 생성
이후 vgg-16 학습 데이터의 fully-connected 모델 학습 후 저장
```

# 참고자료
- [https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/](https://keraskorea.github.io/posts/2018-10-24-little_data_powerful_model/)
- [download vgg16_weights.h5](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)

# 야구 유니폼 분류 테스트

```
python test_classifier_uniform.py 
python test_convert_model.py
python test_model.py
```
1. test_classifier_uniform.py

- training 후 학습된 가중치(weight), layer 구조를 각각 test_uniform_model.h5, test_uniform_model.json으로 저장

2. test_convert_model.py

- 1에서 만들어진 가중치와 json 파일을 이용해 model로 저장함

3. test_model.py

- 2에서 저장한 모델로 예측 테스트

