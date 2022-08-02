from torchvision import models
from torch import nn
import numpy as np
import torch


def initialize_model(model_name, num_classes, feature_extract=True):
    model = None
    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        print(model)

    
    elif model_name == "densenet":
        model = models.densenet161(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        print(model)
    elif model_name == "vgg":
        model = models.vgg19(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        print(model)

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

            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            print(model)

        elif model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
            print(model)

    elif model_name == "resnet":
        model = models.resnet50(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(model)

    elif model_name == "squeezenet":
        model = models.squeezenet1_0(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model.num_classes = num_classes
        print(model)
    
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        print(model)
    
    elif model_name == "resnext":
        model = models.resnext101_64x4d(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(model)

    elif model_name == "googlenet":
        model = models.googlenet(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(model)

    elif model_name == "mnasnet":
        model = models.mnasnet1_0(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        print(model)

    elif model_name == "regnet":
        model = models.regnet_x_16gf(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(model)
    
    elif model_name == "shufflenet":
        model = models.shufflenet_v2_x2_0(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        print(model)

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

        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        print(model)
    
    elif model_name == "ViT":
        model = models.vit_b_16(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        print(model)

    elif model_name == "Swin":
        model = models.swin_t(pretrained=True)
        set_parameter_requires_grad(model, feature_extract)
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
        print(model)
    
    else:
        print("Invalid model name")
        exit()
    
    # Gather the parameters to be optimized/updated in this run.
    params_to_update = set_params_to_update(model, feature_extract)
    
    
    return model, params_to_update


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