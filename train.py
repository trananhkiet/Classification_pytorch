import os
import torch
import argparse
from torch import nn
from torch import optim
from torch.nn import CrossEntropyLoss
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from utils.simple_tools import *
from utils.choose_class_weight import *
from utils.imbalanced import ImbalancedDatasetSampler
from utils.loadjsonconfig import LoadJsonConfig
from utils.models import initialize_model, train_model
from utils.prepare_dataset import MyDataset, make_datapath_list
from utils.image_transform import Image_transform

torch.cuda.empty_cache()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate')
    parser.add_argument('--jsonconfig_path', type=str, help='path to json file')
    parser.add_argument('--model_name', type=str, help='model name')
    args = parser.parse_args()

    jsonconfig_path = args.jsonconfig_path
    model_name = args.model_name

    print("*** CLASSIFICATION MODEL TRAINING ...")
    torch.multiprocessing.freeze_support()

    config = LoadJsonConfig(jsonconfig_path)

    specific_gpu = config.GPU

    device = torch.device("cuda:"+specific_gpu if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print(torch.cuda.is_available())

    logs_dir = config.LOGS_DIR
    model_path = os.path.join(logs_dir, "model")
    log_dir = os.path.join(logs_dir, "LOG")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_list = make_datapath_list(root_path=config.DATASET_PATH, phase='train')
    val_list = make_datapath_list(root_path=config.DATASET_PATH, phase='val')

    train_dataset = MyDataset(train_list, transform=Image_transform(config), phase='train', classnames=config.CLASS_NAME)
    val_dataset = MyDataset(val_list, transform=Image_transform(config), phase='val', classnames=config.CLASS_NAME)


    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, shuffle=False)

    dataloader_dict = {'train': train_dataloader, 'val': val_dataloader}
    
    model = initialize_model(model_name=model_name, num_classes=len(config.CLASS_NAME))
    model = model.train()

    if config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.WEIGHT_DECAY, amsgrad=False)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=config.MOMENTUM)

    writer = SummaryWriter(log_dir+'/')
    epoch = step = 0

    criterion = nn.CrossEntropyLoss()
    train_model(model, model_path, dataloader_dict, criterion=criterion, optimizer=optimizer, num_epochs=config.NO_EPOCH, writer=writer, device=device)



# /home/finn/Classification_pytorch/venv/bin/activate
# python train.py --jsonconfig_path "utils/config.json" --model_name="mobilenet_v2"