import numpy as np
from torchvision import transforms
import albumentations as album


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


class Image_transform:
    def __init__(self, config):
        self.config = config

        self.data_transform = {
            'train': transforms.Compose([
                SquarePad(image_size=self.config.PADDING_SIZE),
                transforms.Resize(self.config.RESIZE),
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