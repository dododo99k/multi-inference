import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class EarlyExitResNet50(nn.Module):
    def __init__(self, original_model, num_classes=1000):
        super(EarlyExitResNet50, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        # Freeze the features part
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Early exit branch
        self.early_exit = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # Conv layer with 512 output channels
            nn.BatchNorm2d(512),                            # BatchNorm layer
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Early exit logic
        early_exit_output = self.early_exit(x)
        
        # Continue with the remaining layers
        # main_exit_output = self.main_exit(x)
        
        return early_exit_output# , main_exit_output

class EarlyExitResNet101(nn.Module):
    def __init__(self, original_model, num_classes=1000):
        super(EarlyExitResNet101, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-3])
        # Freeze the features part
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Early exit branch
        self.early_exit = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # Conv layer with 512 output channels
            nn.BatchNorm2d(512),                            # BatchNorm layer
            nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Early exit logic
        early_exit_output = self.early_exit(x)
        
        # Continue with the remaining layers
        # main_exit_output = self.main_exit(x)
        
        return early_exit_output# , main_exit_output


class EarlyExitMobileNet(nn.Module):
    def __init__(self, original_model, exit_layer = -5, num_classes=10):
        super(EarlyExitMobileNet, self).__init__()
        # expand MobileNet
        detail_layers = []
        for i, child in enumerate(list(original_model.children())):
            for j, sub_child in enumerate(list(child.children())):
                detail_layers.append(sub_child)

        self.features = nn.Sequential(*detail_layers[:exit_layer])
        # self.features = nn.Sequential(*list(original_model.children())[:-1])
        exit_channel = detail_layers[exit_layer-1].out_channels
        # print(f'exit_channel: {exit_channel}')
        # print(self.features)
        
        # Freeze the features part
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Early exit branch
        # self.early_exit = nn.Sequential(*detail_layers[exit_layer-1:])
        # self.early_exit = nn.Sequential(*list(original_model.children())[-1])
        # print(num_classes)
        self.early_exit = nn.Sequential(
            nn.Conv2d(exit_channel, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False),  # Conv layer with 512 output channels
            nn.BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),                            # BatchNorm layer
            nn.ReLU6(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        # print('After fc:', x.size())
        # Early exit logic
        early_exit_output = self.early_exit(x)
        
        # Continue with the remaining layers
        # main_exit_output = self.main_exit(x)
        
        return early_exit_output# , main_exit_output