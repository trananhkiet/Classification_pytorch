import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
import numpy as np
from tensorboardX import SummaryWriter
from utils.simple_tools import *
from numpy import random
from utils.imbalanced import ImbalancedDatasetSampler
from torchsummary import summary
from utils.utils import *
import cv2
import albumentations as album
from PIL import Image

torch.cuda.empty_cache()


class ImageFolder(Dataset):
    def __init__(self, root_dir, transform=None, total_classes=None):
        self.transform = transform
        self.data = []
        
        if total_classes:
            self.classnames  = os.listdir(root_dir)[:total_classes] # for test
        else:
            self.classnames = os.listdir(root_dir)
            
        for index, label in enumerate(self.classnames):
            root_image_name = os.path.join(root_dir, label)
            
            for i in os.listdir(root_image_name):
                full_path = os.path.join(root_image_name, i)
                self.data.append((full_path, index))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data, target = self.data[index]
        img = np.array(Image.open(data))
        
        if self.transform:
            augmentations = self.transform(image=img)
            img = augmentations["image"]
        
        target = torch.from_numpy(np.array(target))
        img = torch.from_numpy(img)
        
        print(type(img),img.shape, target)
            
        
        return img,target 

class SquarePad:
	def __init__(self, image_size = 400):
		self.image_size= image_size

	def __call__(self, image):
		if image.size[0] >= image.size[1]:
			self.image_size = image.size[0]
		else:
			self.image_size = image.size[1]

		p_top = (self.image_size - image.size[1])//2
		p_bottom = self.image_size - image.size[1] - p_top

		p_right = (self.image_size - image.size[0])//2
		p_left = self.image_size - image.size[0] - p_right

		padding = (p_left, p_top, p_right, p_bottom)

		image = transforms.functional.pad(image, padding, 0, 'constant')
		return image

class Convert_Rgb:
	def __call__(self, image):
		image = image.convert("RGB")
		return image

class GrayScale:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, image):
        if random.rand() > self.p:
            image = transforms.functional.to_grayscale(image, 3)
        return image


if __name__ == '__main__':

    torch.multiprocessing.freeze_support()

    config = {
        "epochs": 10,
        "batch_size": 8,
        "logs_dir": "logs",
        "dataset_path": "/home/pevis/TOMO/Datav3",
        "drop_out": 0.2,
        "weight_decay": 0.0001,
        "leaning_rate": 0.0001,
        "num_workers": 2,
        # "pass_threshold": 0.7,
        # "fail_threshold": 0.3,
        "classes": ['Bottle', 'Bottle_cap', 'Bottle_uncap', 'No_bottle'],
        "use_class_weight": True

    }
    
    device = torch.device("cuda:3")

    logs_dir = config["logs_dir"]
    model_path = os.path.join(logs_dir,"model")
    log_dir = os.path.join(logs_dir,"LOG")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    

    anbu = album.Compose([

        #Convert_Rgb(),

        # SquarePad(), #them pixel 0 de thanh hinh vuong
        # transforms.Resize(512),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.5),
        # transforms.GaussianBlur(3, sigma=(0.1, 2.0), p=0.5),
        album.Resize(512, 512),
        album.HorizontalFlip(p=0.2),
        album.VerticalFlip(p=0.2),
        album.GaussianBlur(blur_limit=(3, 7), p=0.2),
        album.RandomRain(brightness_coefficient=0.7, drop_width=1, blur_value=5, p=0.2),
        album.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=0.2),
        album.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.2),
        album.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.2),
        album.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
    
    


        #GrayScale(),

    ])
    def albu(image, transform=anbu):

        image = image.convert("RGB")
        image = np.array(image)

        if transform:
            image =  transform(image=image)["image"]
        return image

    train_transforms = transforms.Compose([
        SquarePad(), # them pixel 0 de thanh hinh vuong
        # transforms.Resize(512),
        transforms.Lambda(albu),
        #GrayScale(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        SquarePad(),
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.409],[0.229, 0.224, 0.225])
    ])



    dataset = datasets.ImageFolder(os.path.join(config["dataset_path"], 'Train'), transform=train_transforms)
    #dataset = ImageFolder(os.path.join(config["dataset_path"], 'Train'), total_classes=4,transform=train_transforms)
    data_val = datasets.ImageFolder(os.path.join(config["dataset_path"], 'Test'), transform=test_transforms)
    dataloader = torch.utils.data.DataLoader(
		dataset,
		sampler = ImbalancedDatasetSampler(dataset), 
		batch_size=config['batch_size'], 
        #shuffle=True,
		num_workers=config["num_workers"])
    dataloader_val = torch.utils.data.DataLoader(
        data_val,
        #sampler = ImbalancedDatasetSampler(data_val),
        batch_size=1,
        #shuffle=True,
        num_workers=config["num_workers"])

    

    #Model
    model_ft = models.mobilenet_v2(pretrained=True)
    #phai su dung code nay de thay the
    num_ftrs = model_ft.classifier[1].in_features
    model_ft.classifier = nn.Sequential(
        nn.Dropout(p=config["drop_out"], inplace=False),
        nn.Linear(in_features=num_ftrs, out_features=4, bias=True)
    )

    # model_ft.classifier[-1] = nn.Linear(1280, 4)
    # )
    #model_ft.load_state_dict(torch.load('log_model_pretrain_adam\model\model_1000\model.pth'))
    model_ft = model_ft.to(device)
    
    # print(model_ft)
    #summary(model_ft, (3, 512, 512))

    # Optimizer
    #optimizer = optim.SGD(model_ft.parameters(), lr=1e-4, weight_decay=4e-5, momentum=0.9)
    optimizer = optim.Adam(model_ft.parameters(), lr=config["leaning_rate"], betas=(0.9, 0.999), eps=1e-08, weight_decay=config['weight_decay'], amsgrad=False)
    # #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    writer = SummaryWriter(log_dir+'/')

    epoch = step = 0

    if config['use_class_weight']:
        cls_weight = choose_class_weight(os.path.join(config["dataset_path"], 'Train'), config["classes"])
        print("using class weight", cls_weight)
        cls_weight_val = choose_class_weight(os.path.join(config["dataset_path"], 'Test'), config["classes"])
        print("using class weight val", cls_weight_val)
        cls_weight = torch.FloatTensor(cls_weight).to(device)
        cls_weight_val = torch.FloatTensor(cls_weight_val).to(device)
    else:
        cls_weight = None
        cls_weight_val = None

    
    crossentropy = CrossEntropyLoss(weight=cls_weight)
    crossentropy_val = CrossEntropyLoss(weight=cls_weight_val)
    while epoch < config['epochs']:
        list_loss = []
        for img, label in iter(dataloader):
            model_ft.train()

            img = img.to(device)
            label = label.to(device)
            #print(label)

            optimizer.zero_grad()
            theta = (model_ft(img))	

            loss = crossentropy(theta, label)

            loss.backward()
            optimizer.step()
            list_loss.append(loss)
            
            

            print(f"Global Epoch: {epoch} ==== Loss: {loss}")
        mean_loss = sum(list_loss)/len(list_loss)
        model_ft.eval()
        with torch.no_grad():
            val = val_loss(dataloader_val, model_ft, device, crossentropy_val) # From simple_tools
            acc_train = accuracy(dataloader, model_ft, device)
            acc_val = accuracy(dataloader_val, model_ft, device)
            

            torch.save(model_ft.state_dict(), model_path + '/' + str(epoch) + '.pth')
        print(val, acc_train, acc_val)
        writer.add_scalars('Loss',{"Train": mean_loss, "Val": val}, epoch)
        writer.add_scalars('Accuracy',{"Train": acc_train, "Val": acc_val}, epoch)


        print("Evaluate model: ........")
        #exp_lr_scheduler.step()
        epoch += 1

    print("end")