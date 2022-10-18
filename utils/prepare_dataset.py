from PIL import Image
import torch.utils.data as data
import glob
import os


def make_datapath_list(root_path, phase='train'):
    rootpath = root_path
    path_list = glob.glob(os.path.join(rootpath, phase, '*/*.png')) + glob.glob(os.path.join(rootpath, phase, '*/*.bmp'))
    
    return path_list

class MyDataset(data.Dataset):
    def __init__(self, file_list, transform=None, phase='train', classnames =[]):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.classnames = classnames
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        
        img_transformed = self.transform(img, self.phase)
        
        if self.phase == 'train':
            label = img_path.split('\\')[-2]
        elif self.phase == 'val':
            label = img_path.split('\\')[-2]
        elif self.phase == 'test':
            label = img_path.split('\\')[-2]

        for index, lbl in enumerate(self.classnames):
            if label == lbl:
                label = index
        
        return img_transformed, label
