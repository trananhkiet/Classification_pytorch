import torch
import argparse
from torchvision import datasets, transforms
from utils.data_generator import *
import glob
import os
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
from utils.loadjsonconfig import LoadJsonConfig
from utils.models import initialize_model
import io 
import xlsxwriter
from prettytable import PrettyTable
from tqdm import tqdm

def load_model(net, model_path):
    load_weights = torch.load(model_path, map_location=torch.device('cpu'))
    net.load_state_dict(load_weights)
    return net

class Predictor():
    def __init__(self, classnames, model_name, model_path, num_classes):
        self.class_dict = classnames
        model = initialize_model(model_name=model_name, num_classes=num_classes)
        model.eval()
        # prepare model
        self.model = load_model(model, model_path=model_path)

    def predict_max(self, output_net, list_predict):
        max_id = np.argmax(output_net.detach().numpy())
        list_predict.append(max_id)
        predicted_label = self.class_dict[max_id]
        return predicted_label, list_predict

    def predict(self, img, list_predict, test_transforms):
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
    parser.add_argument('--model_name', type=str, help='model name')
    parser.add_argument('--model_path', type=str, help='path to model')
    args = parser.parse_args()

    jsonconfig_path = args.jsonconfig_path
    model_name = args.model_name
    model_path = args.model_path

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

    # data_test = datasets.ImageFolder(os.path.join(config.DATASET_PATH, 'test'), transform=test_transforms)
    # dataloader_test = torch.utils.data.DataLoader(
    #     data_test,
    #     batch_size=1,
    #     num_workers=config.NUM_WORKERS
    # )

    # y_pred = []
    # y_true = []
    # for _, label in iter(dataloader_test):
    #     y_true.extend(label.numpy())

    list_filename = glob.glob(config.DATASET_PATH+"/test"+"/*/*.png")
    list_predict = []
    predictor = Predictor(classnames=classnames, model_name=model_name, model_path=model_path, num_classes=len(config.CLASS_NAME))

    dataset_name = config.DATASET_PATH.split(os.path.sep)[-1]
    report_file_name = model_name + '_' + dataset_name + '.xlsx'
    image_display_size = int(config.RESIZE * 1.5)
    name_width = len(os.path.splitext(os.path.basename(list_filename[0]))[0]) + 10 
    label_width = len(max(classnames, key=len)) 

    workbook = xlsxwriter.Workbook(report_file_name)
    worksheet = workbook.add_worksheet(dataset_name)

    worksheet.set_column("B:B", name_width)  # image name column
    worksheet.set_column("C:C", 25)          # image column
    worksheet.set_column("D:E", label_width) # label columns
    worksheet.set_column("F:I", 15)          # score columns
    worksheet.set_row(0, height=16)

    cell_format = workbook.add_format()
    cell_format.set_align('center')
    cell_format.set_align('vcenter')
    cell_format.set_text_wrap()

    highlight_format = workbook.add_format()
    highlight_format.set_align('center')
    highlight_format.set_align('vcenter')
    highlight_format.set_bg_color("red")

    Header = ["Image_id", "Image", "Label", "Predict"]
    Header.extend(classnames)
    Header.append("Result")
    start_row, start_column = 0, 1
    worksheet.write_row(start_row, start_column, Header, cell_format)
    
    def resize_image_data(image_path, bound_width_height):
        """ Get the image path, resize image and return image data in byte format """
        im = Image.open(image_path)
        im.thumbnail(bound_width_height, Image.LANCZOS)  # LANCZOS is important for shrinking, use Image.ANTIALIAS if LANCZOS does not work
        im_bytes = io.BytesIO()
        im.save(im_bytes, format='PNG')
        return im_bytes

    y_true = []
    for i in tqdm(range(len(list_filename))):
        y_true_i = os.path.basename(os.path.dirname(list_filename[i]))
        y_true.append(classnames.index(y_true_i))
        img = Image.open(list_filename[i])
        img = img.convert("RGB")

        res, output, y_pred = predictor.predict(img, list_predict, test_transforms)
        image_data = resize_image_data(list_filename[i], (image_display_size, image_display_size))

        Data = [0] * len(Header)
        Data[0] = os.path.splitext(os.path.basename(list_filename[i]))[0] # image name
        # Data[1] : img 
        Data[2] = classnames[y_true[i]] # label ground truth of image 
        Data[3] = res # predict result of image 
        Data[4:4+len(classnames)] = output.tolist()[0] # 
        Data[-1] = classnames[y_true[i]] == res # 

        start_row += 1
        worksheet.set_row(start_row, height=120)

        for index, info in enumerate(Data):
            excel_format = highlight_format if (Data[index] == False and isinstance(Data[index], bool)) else cell_format
            if index == 1:
                worksheet.insert_image(start_row, start_column + index, list_filename[i],
                                       {'x_scale': 0.5, 'y_scale': 0.4, 'x_offset': 5, 'y_offset': 5,
                                        'object_position': 1, 'image_data': image_data})
            else:
                worksheet.write(start_row, start_column + index, Data[index], excel_format)

    column_headers = [{'header': head} for head in Header]
    worksheet.add_table(0, 1, start_row, len(Header), {'columns': column_headers})
    worksheet.freeze_panes(1, 0)
    worksheet.hide_gridlines(2)
    workbook.close()

    cf_matrix = confusion_matrix(y_true, y_pred)
    try: # print confusion matrix table
        myTable = PrettyTable(["-"] + classnames)
        for index in range(len(classnames)):
            myTable.add_row([classnames[index]] + cf_matrix[index].tolist())
        print(f"Confusion matrix : \n{myTable}")
    except:
        print(f"Confusion matrix : \n{cf_matrix}")