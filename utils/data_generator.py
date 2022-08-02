from ctypes import resize
from numpy import random
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


anbu = album.Compose([
        album.Resize(512, 512),
        # album.HorizontalFlip(p=0.2),
        # album.VerticalFlip(p=0.2),
        # album.GaussianBlur(blur_limit=(3, 7), p=0.2),
        # album.RandomRain(brightness_coefficient=0.7, drop_width=1, blur_value=5, p=0.2),
        # album.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=0.2),
        # album.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=0.2),
        # album.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=0.2),
        # album.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.2),
        #GrayScale(),

    ])
def albu(image, transform=anbu):
    image = image.convert("RGB")
    image = np.array(image)
    if transform:
        image = transform(image=image)["image"]
    return image

