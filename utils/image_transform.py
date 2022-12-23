import numpy as np
from torchvision import transforms
import cv2
import albumentations as A
import random
import numpy as np
from PIL import Image


class SquarePad:
    def __init__(self, image_size=512):
        self.image_size=image_size

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
    
class Augmentation():
    def __init__(self, augment_list=None, random_mode = False, input_image_size = None):

        self.available_augment = [
                                  "random_brightness", 
                                  "shuffle_channel", 
                                  "to_gray", 
                                  "rgb_shift", 
                                  "huesaturationvalue",
                                  "clahe",
                                  "random_gamma",
                                  "blur",
                                  "median_blur",
                                  ]
        self.augment_list = augment_list
        self.random_mode = random_mode
        self.input_image_size = input_image_size
        
    
    def get_augment_list(self):
        return self.available_augment

    def get_transform(self):
        augment_func = []
        if self.random_mode:
            augment_choice = self.random_augment()
            augment_func.append(augment_choice)
            return augment_func
        else:
            if self.augment_list is not None:
                for _augment_name in self.augment_list:
                    augment_func.append(self.get_augment(_augment_name))
                return augment_func


    def augment_function(self, image):

        transform_list = self.get_transform()
        train_transforms = A.Compose(transform_list)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image = train_transforms(image=image)['image']
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image, 'RGB')

    def random_augment(self):
        
        random_augment = random.choice(self.available_augment)
        augment_choice = self.get_augment(random_augment)

        return augment_choice

    def get_augment(self, augment_name):

        if augment_name == "random_crop": 
            return A.RandomResizedCrop(self.input_image_size, self.input_image_size, scale=(0.8, 1.0), ratio=(0.7, 1.35))

        elif augment_name == "random_brightness":
            return A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.3)

        elif augment_name == "shuffle_channel":
            return A.ChannelShuffle(p=0.5)

        elif augment_name == "to_gray":
            return A.ToGray(p=0.5)
            
        elif augment_name == "rgb_shift":
            return A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, always_apply=False, p=0.5)

        elif augment_name == "huesaturationvalue":
            return A.HueSaturationValue (hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5)

        elif augment_name == "clahe":
            return A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5)

        elif augment_name == "random_gamma":
            return A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5)

        elif augment_name == "blur":
            return A.Blur(blur_limit=7, always_apply=False, p=0.5)

        elif augment_name == "median_blur":
            return A.MedianBlur(blur_limit=7, always_apply=False, p=0.5)

        elif augment_name == "imagecompression":
            return A.ImageCompression(quality_lower=99, quality_upper=100, always_apply=False, p=0.5)

        else:

            raise Exception("Augment not available")


class Image_transform:
    def __init__(self, config):
        self.config = config
        
        training_transforms_func = Augmentation(random_mode = True, input_image_size = self.config.RESIZE).augment_function
        self.data_transform = {
            'train': transforms.Compose([
                SquarePad(image_size=self.config.PADDING_SIZE),
                transforms.Resize(self.config.RESIZE),
                transforms.Lambda(training_transforms_func),
                transforms.ToTensor(),
                transforms.Normalize(self.config.MEAN, self.config.STD)
            ]),
            'val': transforms.Compose([
                SquarePad(image_size=self.config.PADDING_SIZE),
                transforms.Resize(self.config.RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(self.config.MEAN, self.config.STD)
            ]),
            'test': transforms.Compose([
                SquarePad(image_size=self.config.PADDING_SIZE),
                transforms.Resize(self.config.RESIZE),
                transforms.ToTensor(),
                transforms.Normalize(self.config.MEAN, self.config.STD)
            ])
        }
    
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)