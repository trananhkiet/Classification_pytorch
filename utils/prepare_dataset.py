from PIL import Image
import torch.utils.data as data
import glob
import os
from pathlib import Path
import os.path


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
        img = Image.open(img_path).convert('RGB')

        img_transformed = self.transform(img, self.phase)

        dir_name = Path(img_path).parents[0]
        label = os.path.basename(dir_name)
        
        index_label = self.classnames.index(label)
        
        if label not in self.classnames:
            raise Exception(f"Label {label} not in classname!")
        
        return img_transformed, index_label
