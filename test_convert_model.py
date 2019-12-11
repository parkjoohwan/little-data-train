from tensorflow.keras.models import model_from_json

# test_claasifier_uniform.py 실행 결과로 생긴
# test_uniform_model.json(layer 구조 정보), test_uniform_model.h5(가중치) 파일을
# model로 저장함

json_file = open('test_uniform_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights('test_uniform_model.h5')

loaded_model.save('uniform_model.h5')
