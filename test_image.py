import imp
from eval import Predictor
from utils.loadjsonconfig import LoadJsonConfig
from PIL import Image


jsonconfig_path = "utils/config.json"
model_name = "mobilenet_v2"
model_path = "D:/Classification_Module/Classification_pytorch/epoch_72.pth"

config = LoadJsonConfig(jsonconfig_path) 


input_image = Image.open("Label_classification/test/finger_no_labeled/103background_13_9_2.png")
input_image = input_image.convert("RGB")
predictor = Predictor(config, model_name=model_name, model_path=model_path)
idx, result = predictor.predict_image(input_image)

# predictor.export_excel()
print("class_index:", idx)
print( "class_name:", result)