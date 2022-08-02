import torch
import argparse
from torchvision import datasets
from utils.data_generator import *
import glob
import os
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
from utils.loadjsonconfig import LoadJsonConfig
import utils.models as models

def load_model(net, model_path):
    load_weights = torch.load(model_path)
    net.load_state_dict(load_weights)
    return net

class Predictor():
    def __init__(self, classnames, model_name, model_path, num_classes):
        self.class_dict = classnames

        model, params_to_update = models.initialize_model(model_name=model_name, num_classes=num_classes)
        model.eval()
        # prepare model
        self.model = load_model(model, model_path=model_path)

    def predict_max(self, output_net, list_predict):
        max_id = np.argmax(output_net.detach().numpy())
        list_predict.append(max_id)
        predicted_label = self.class_dict[max_id]
        return predicted_label, list_predict

    def predict(self, img, list_predict):
        # prepare network
        img = test_transforms(img)
        img = img.unsqueeze_(0)  # (channel, height, width) -> (1, channel, height, width)

        # predict
        output = self.model(img)
        output = torch.nn.Softmax(dim=1)(output.detach().cpu())
        response, list_pred = self.predict_max(output_net=output, list_predict=list_predict)
        return response, output, list_pred

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--jsonconfig_path', type=str, help='path to json file')
    parser.add_argument('--num_classes', type=int, help='number of classes')
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--testset_path', type=str, help='path to testset')
    args = parser.parse_args()

    jsonconfig_path = args.jsonconfig_path
    num_classes = args.num_classes
    model_name = args.model_name
    model_path = args.model_path
    testset_path = args.testset_path

    config = LoadJsonConfig(jsonconfig_path)

    classnames = config.CLASS_NAME
    torch.multiprocessing.freeze_support()
    device = torch.device("cpu")

    test_transforms = transforms.Compose([
        SquarePad(image_size=config.PADDING_SIZE),
        transforms.Resize(config.RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    data_test = datasets.ImageFolder(os.path.join(config.DATASET_PATH, 'test'), transform=test_transforms)
    dataloader_test = torch.utils.data.DataLoader(
        data_test,
        batch_size=1,
        num_workers=config.NUM_WORKERS
    )

    y_pred = []
    y_true = []
    for _, label in iter(dataloader_test):
        y_true.extend(label.numpy())

    list_filename = glob.glob(testset_path+"/*/*.png")
    list_predict = []
    predictor = Predictor(classnames=classnames, model_name=model_name, model_path=model_path, num_classes=num_classes)
    for i in range(len(list_filename)):
        img = Image.open(list_filename[i])
        img = img.convert("RGB")
        res, output, y_pred = predictor.predict(img, list_predict)

    cf_matrix = confusion_matrix(y_true, y_pred)
    print(cf_matrix)
