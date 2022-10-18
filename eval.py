import torch
import argparse
from torchvision import datasets, transforms
import glob
import os
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
from utils.loadjsonconfig import LoadJsonConfig
from utils.models import initialize_model
from utils.image_transform import Image_transform
import io 
import xlsxwriter
from prettytable import PrettyTable
from tqdm import tqdm

def load_model(net, model_path):
    load_weights = torch.load(model_path, map_location=torch.device('cpu'))
    net.load_state_dict(load_weights)
    return net

class Predictor():
    def __init__(self, config, model_name, model_path):
        self.model_name = model_name
        self.config = config
        self.class_dict = self.config.CLASS_NAME
        model = initialize_model(model_name=model_name, num_classes=len(self.config.CLASS_NAME))
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
        transform = Image_transform(self.config)
        img = transform(img, phase='test')
        img = img.unsqueeze_(0)  # (channel, height, width) -> (1, channel, height, width)

        # predict
        output = self.model(img)
        output = torch.nn.Softmax(dim=1)(output.detach().cpu())
        response, list_pred = self.predict_max(output_net=output, list_predict=list_predict)
        return response, output, list_pred

    def predict_image(self, img):
        transform = Image_transform(self.config)
        img = transform(img, phase='test')
        img = img.unsqueeze_(0)

        output = self.model(img)
        output = torch.nn.Softmax(dim=1)(output.detach().cpu())
        max_id = np.argmax(output.detach().numpy())
        predicted_label = self.class_dict[max_id]
        return max_id, predicted_label
        

    def export_excel(self):
        list_predict = []
        list_filename = glob.glob(self.config.DATASET_PATH+"/test"+"/*/*.png")
        dataset_name = self.config.DATASET_PATH.split(os.path.sep)[-1]
        report_file_name = self.model_name + '_' + dataset_name + '.xlsx'
        image_display_size = int(self.config.RESIZE * 1.5)
        name_width = len(os.path.splitext(os.path.basename(list_filename[0]))[0]) + 10 
        label_width = len(max(self.config.CLASS_NAME, key=len)) 

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
        Header.extend(self.config.CLASS_NAME)
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
            y_true.append(self.config.CLASS_NAME.index(y_true_i))
            img = Image.open(list_filename[i])
            img = img.convert("RGB")

            res, output, y_pred = self.predict(img, list_predict)
            image_data = resize_image_data(list_filename[i], (image_display_size, image_display_size))

            Data = [0] * len(Header)
            Data[0] = os.path.splitext(os.path.basename(list_filename[i]))[0] # image name
            # Data[1] : img 
            Data[2] = self.config.CLASS_NAME[y_true[i]] # label ground truth of image 
            Data[3] = res # predict result of image 
            Data[4:4+len(self.config.CLASS_NAME)] = output.tolist()[0] # 
            Data[-1] = self.config.CLASS_NAME[y_true[i]] == res # 

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
            myTable = PrettyTable(["-"] + self.config.CLASS_NAME)
            for index in range(len(self.config.CLASS_NAME)):
                myTable.add_row([self.config.CLASS_NAME[index]] + cf_matrix[index].tolist())
            print(f"Confusion matrix : \n{myTable}")
        except:
            print(f"Confusion matrix : \n{cf_matrix}")
        

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
    
    predictor = Predictor(config, model_name=model_name, model_path=model_path)

    predictor.export_excel()

    