from keras.models import model_from_json, model_from_yaml, load_model

json_string = model.to_json()
yaml_string = model.to_yaml()

mod = model_from_json(json_string)
mod = model_from_yaml(yaml_string)

mod.save('my_model.h5')
mod = load_model('my_model.h5')
