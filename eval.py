from tkinter import image_names
import torch
import torchvision
from torchvision import datasets, transforms, models
from utils.utils import *
from torch import nn

import matplotlib.pyplot as plt
import cv2 
import time
import glob
from sklearn.metrics import confusion_matrix
import pandas as pd 
from PIL import Image
import seaborn as sn
import numpy as np
import albumentations as album
from albumentations.pytorch import ToTensorV2

class SquarePad:
    def __init__(self,image_size = 400):
        self.image_size= image_size

    def __call__(self, image):

        p_top = (self.image_size - image.size[1])//2
        p_bottom = self.image_size - image.size[1] - p_top

        p_right = (self.image_size - image.size[0])//2
        p_left = self.image_size - image.size[0] - p_right

        padding = (p_left, p_top, p_right, p_bottom)
        image = transforms.functional.pad(image, padding, 0, 'constant')

        return image

if __name__ == '__main__':

    torch.multiprocessing.freeze_support()

    device = torch.device("cpu")

    # transform = transforms.Compose([
    #     SquarePad(512), # Tăng kích thước ảnh lên 512 bằng cách thêm pixel = 0      
    #     transforms.Resize(512), # Phải để trước toTensor vì type PIL mới resize đc 
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])

    anbu = album.Compose([

        #Convert_Rgb(),

        # SquarePad(), #them pixel 0 de thanh hinh vuong
        # transforms.Resize(512),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(3, sigma=(0.1, 2.0), p=0.5),
        album.Resize(512, 512),
        #album.HorizontalFlip(p=0.15),
        #album.VerticalFlip(p=0.15),
        #album.GaussianBlur(blur_limit=(3, 7), p=0.15),
        #album.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=0.15),
        #album.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=0.15),
        #album.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.15),
        #album.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.15),
        #album.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.15),
    
    


        #GrayScale(),
        # ToTensorV2(),
        # album.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    ])
    anbu2 = album.Compose([
        album.Resize(512, 512),

        album.HorizontalFlip(p=0.15),
        # album.VerticalFlip(p=0.15),
        # album.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=0.15),
        # album.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=0.15),
    ])

    def albu(image, transform=anbu):

        image = image.convert("RGB")
        image = np.array(image)

        if transform:
            image =  transform(image=image)["image"]
        return image
    
    
    def albu2(image, transform=anbu2):

        image = image.convert("RGB")
        image = np.array(image)

        if transform:
            image =  transform(image=image)["image"]
        return image
    

    transform = transforms.Compose([
        SquarePad(), # Tăng kích thước ảnh lên 512 bằng cách thêm pixel = 0      
        transforms.Lambda(albu), # Phải để trước toTensor vì type PIL mới resize đc 
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform2 = transforms.Compose([
        SquarePad(), # Tăng kích thước ảnh lên 512 bằng cách thêm pixel = 0      
        transforms.Lambda(albu2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #transform = transforms.Compose([SquarePad(),transforms.Resize((128, 128))])

    #data_eval = datasets.ImageFolder('../splitted_24-12-FAnoel/Validation/', transform=transform)
    #dataloader_eval = torch.utils.data.DataLoader(data_eval, batch_size=16, shuffle=True, num_workers=2)

    #model = models.googlenet(num_classes = 4, init_weights=True)
    #model.fc = nn.Linear(1024, 4)
    #model.to(device)
    #model_ft=models.mobilenet_v3_large(pretrained=True)
#
    model = models.mobilenet_v2(num_classes = 4).to(device)
    # model.load_state_dict(torch.load('/home/jay2/CONTACT_LENS/MobileNet/logs_model2.0/model/model_35/model.pth'))
    #model.load_state_dict(torch.load(r"D:\log\log3\New folder\40.pth", map_location='cpu'), strict=False)
    model.load_state_dict(torch.load(r"D:\log\16.pth", map_location='cpu'))
      
    model.eval()

    data_eva = r'D:\rgb\2022_05_10\rgb\*' 
    #wrong_path = '/home/pevis/TOMO/code/wrong_class1'

    img_path = glob.glob(data_eva)
    print(len(img_path))

    y_pred = []
    y_true = []
    list_score=[]
    ori_img=[]
    ori_img2=[]
    list_m=[]
    with torch.no_grad():
        print("predicting...")
        for img_name in img_path:
            #img_path = os.path.join(test_folder, img_name)
            
            #img = Image.open(img_name)
            #print(img)
            #print(img.size)
            # print(img_name)
            img = cv2.imread(img_name)
            # print(img.shape)
            r = cv2.selectROI(img, False)
            img_crop = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            img_crop2 = img[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            img = Image.fromarray(img_crop)
            img2 = Image.fromarray(img_crop2)
            
            
            
            # print(img)
            ori_img.append(np.asarray(img))
            ori_img2.append(np.asarray(img2))
            img = transform(img)
            img2 = transform2(img2)
            img = torch.unsqueeze(img, 0)   
            img2 = torch.unsqueeze(img2, 0) 
            img_pre = img.to(device)
            img_pre2 = img2.to(device)
            
            start_time = time.time()
            #img = img.to(device).unsqueeze(0)
            score = model(img_pre).cpu()
            score2 = model(img_pre2).cpu()

            score3 = score + score2

            score = torch.nn.Softmax(dim=1)(score).numpy()  

            
            #     if score[0][0] < 0.5:
            #         score[0][0] = 0
            #for i in range(score.shape[0]): 
            # if score[0][0] < 0.7:
            #     print(f"----------{score[0][0]}---------")
            #     score[0][0] = 0
            # if score[0][3] < 0.92:
            #     print(f"----------{score[0][3]}---------")
            #     score[0][3] = 0
            
            y_pred.append(np.argmax(score))  #extend có thể chèn list vào sau list
            list_score.append(score)
            m = np.argmax(score)
            # list_m.append(m)
            # if 'Bottle_uncap' in img_name:
            #     y_true.append(2)
            # elif 'No_bottle' in img_name:
            #     y_true.append(3)
            # elif 'Bottle_cap' in img_name:
            #     y_true.append(1)
            # else:
            #     y_true.append(0)

            cv2.imwrite(os.path.join("result",str(m) + "_" + str(score[0][m]) + "_" + os.path.basename(img_name)),img_crop)

            print(score)
            # print(score2)
            # print(score3)
            # if m == 0:
            #     print('Bottle')
            # elif m == 1:
            #     print('Bottle_cap')
            # elif m==2:
            #     print('Bottle_uncap')
            # else:
            #     print('No_bottle')

            


                


            #print("Inference time: ", (time.time()-start_time)*1000)

    # for i in range(len(y_true)):
    #     if y_pred[i] != y_true[i]:
            
    #         cv2.imwrite(os.path.join(wrong_path, str(y_pred[i])+'_'+str((list_score[i])[0][list_m[i]])+'.bmp'), cv2.cvtColor(ori_img[i], cv2.COLOR_RGB2BGR))


    # classes = ['Bottle', 'Bottle_cap', 'Bottle_uncap', 'No_bottle']
    # cf_matrix = confusion_matrix(y_true, y_pred)
    # df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
    #                  columns = [i for i in classes], dtype=int)

    # plt.figure(figsize = (12,7))
    # sn.heatmap(df_cm, annot=True, fmt="d")
    # plt.savefig('/home/pevis/TOMO/code/image')