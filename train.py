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
from utils.data_generator import *
from utils.loadjsonconfig import LoadJsonConfig
from utils.models import initialize_model

torch.cuda.empty_cache()

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

    logs_dir = config.LOGS_DIR
    model_path = os.path.join(logs_dir, "model")
    log_dir = os.path.join(logs_dir, "LOG")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_transforms = transforms.Compose([
        SquarePad(image_size=config.PADDING_SIZE),
        transforms.Lambda(albu),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    valid_transforms = transforms.Compose([
        SquarePad(image_size=config.PADDING_SIZE),
        transforms.Resize(config.RESIZE),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD)
    ])

    dataset = datasets.ImageFolder(os.path.join(config.DATASET_PATH, 'train'), transform=train_transforms)
    data_val = datasets.ImageFolder(os.path.join(config.DATASET_PATH, 'val'), transform=valid_transforms)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=ImbalancedDatasetSampler(dataset),
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    dataloader_val = torch.utils.data.DataLoader(
        data_val,
        batch_size=1,
        num_workers=config.NUM_WORKERS
    )
    # Model
    model = initialize_model(model_name=model_name, num_classes=len(config.CLASS_NAME))
    model = model.train()

    if config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.WEIGHT_DECAY, amsgrad=False)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=config.MOMENTUM)

    writer = SummaryWriter(log_dir+'/')
    epoch = step = 0

    if eval(config.USE_CLASS_WEIGHT):
        cls_weight = choose_class_weight(os.path.join(config.DATASET_PATH, 'Train'), config.CLASS_NAME)
        print("using class weight", cls_weight)
        cls_weight_val = choose_class_weight(os.path.join(config.DATASET_PATH, 'Test'), config.CLASS_NAME)
        print("using class weight val", cls_weight_val)
        cls_weight = torch.FloatTensor(cls_weight).to(device)
        cls_weight_val = torch.FloatTensor(cls_weight_val).to(device)
    else:
        cls_weight = None
        cls_weight_val = None

    crossentropy = CrossEntropyLoss(weight=cls_weight)
    crossentropy_val = CrossEntropyLoss(weight=cls_weight_val)
    model.to(device)

    while epoch < config.NO_EPOCH:
        list_loss = []
        for img, label in iter(dataloader):
            model.train()

            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            theta = (model(img))	
            loss = crossentropy(theta, label)

            loss.backward()
            optimizer.step()
            list_loss.append(loss)
            
            print(f"Global Epoch: {epoch} ==== Loss: {loss}")
        mean_loss = sum(list_loss)/len(list_loss)
        model.eval()
        with torch.no_grad():
            val = val_loss(dataloader_val, model, device, crossentropy_val)
            acc_train = accuracy(dataloader, model, device)
            acc_val = accuracy(dataloader_val, model, device)

            torch.save(model.state_dict(), model_path + '/' + str(epoch) + '.pth')
        print(val, acc_train, acc_val)
        writer.add_scalars('Loss', {"Train": mean_loss, "Val": val}, epoch)
        writer.add_scalars('Accuracy', {"Train": acc_train, "Val": acc_val}, epoch)

        print("Evaluate model: ........")
        epoch += 1

    print("end")

# python train.py --jsonconfig_path "config.json" --model_name="mobilenet_v2"