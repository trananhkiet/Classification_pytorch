from torchvision import models
from torch import nn
import numpy as np
import torch
import os

def initialize_model(model_name, num_classes):
    model = None
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        # print(model)

    
    elif model_name == "densenet":
        model = models.densenet161(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        # print(model)
    elif model_name == "vgg":
        model = models.vgg19(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        # print(model)

    elif "resnet" in model_name:
        if model_name.startswith('resnet') and not model_name.startswith('resnet-'):
            if model_name == "resnet34":
                model = models.resnet34(pretrained=True)
            elif model_name == "resnet50":
                model = models.resnet50(pretrained=True)
            elif model_name == "resnet101":
                model = models.resnet101(pretrained=True)
            elif model_name == "resnet152":
                model = models.resnet152(pretrained=True)

            # set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            # print(model)

        elif model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            # set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            # print(model)

    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        # print(model)
    
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        # print(model)
    
    elif model_name == "resnext":
        model = models.resnext101_64x4d(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        # print(model)

    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        # print(model)

    elif model_name == "mnasnet":
        model = models.mnasnet1_0(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        # print(model)

    elif model_name == "regnet":
        model = models.regnet_x_16gf(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        # print(model)
    
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x2_0(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        # print(model)

    elif "efficientnet" in model_name:
        if model_name == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=True)
        elif model_name == "efficientnet_b1":
            model = models.efficientnet_b1(pretrained=True)
        elif model_name == "efficientnet_b2":
            model = models.efficientnet_b2(pretrained=True)
        elif model_name == "efficientnet_b3":
            model = models.efficientnet_b3(pretrained=True)
        elif model_name == "efficientnet_b4":
            model = models.efficientnet_b4(pretrained=True)
        elif model_name == "efficientnet_b5":
            model = models.efficientnet_b5(pretrained=True)
        elif model_name == "efficientnet_b6":
            model = models.efficientnet_b6(pretrained=True)
        elif model_name == "efficientnet_b7":
            model = models.efficientnet_b7(pretrained=True)
        else:
            raise KeyError('Un support %s.' % model_name)

        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        # print(model)
    
    elif model_name == "ViT":
        model = models.vit_b_16(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        # print(model)

    elif model_name == "Swin":
        model = models.swin_t(pretrained=True)
        # set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
        # print(model)
    
    else:
        print("Invalid model name")
        exit()
    
    # Gather the parameters to be optimized/updated in this run.
    # params_to_update = set_params_to_update(model, feature_extract)
    
    
    # return model, params_to_update
    return model

def train_model(net, save_model_path, dataloader_dict, criterion, optimizer, num_epochs):
    # declare device
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # chuyen network vao trong device GPU/CPU
    # net = DataParallel(net,device_ids=[0, 1, 2])
    net.to(device)
    # tang toc do tinh toan
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            if (epoch == 0) and (phase == 'train'):
                continue
            for inputs, labels in dataloader_dict[phase]:
                # chuyen inputs va labels vao trong device GPU/CPU
                inputs = inputs.to(device)
                labels = labels.to(device)

                # set gradient of optimizer to be zero
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)

                    # outputs có sô hàng = batch_size
                    #         có số cột = số lượng class
                    # outputs là matran 4x2
                    # tìm giá trị lớn nhất trong mỗi hàng
                    _, preds = torch.max(outputs, 1)

                    # dùng loss để backward
                    # backward tính đạo hàm loss để updata parameter
                    if phase == 'train':
                        loss.backward()
                        # update optimizer
                        optimizer.step()  # step() để update parameter cho optimizer

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloader_dict[phase].dataset)
            epoch_accuracy = epoch_corrects.double() / len(dataloader_dict[phase].dataset)
            print("{} loss:{:.4f} Acc: {:4f}".format(phase, epoch_loss, epoch_accuracy))
            # if phase == 'val':
            #     lr_scheduler.step(epoch_loss)

        torch.save(net.state_dict(), os.path.join(save_model_path, 'epoch_{}.pth'.format(epoch)))

def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

def set_params_to_update(model, feature_extract):
    params_to_update = []
    if feature_extract:
        for param in model.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
    else:
        params_to_update = model.parameters()
    return params_to_update