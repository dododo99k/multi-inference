import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def check_features(sub_layers):
    for i, module in enumerate(list(sub_layers.modules())):
        try:
            final_features = module.num_features
        except:
            pass
    # print('final_features: ',final_features)
    return final_features

def calculate_loss(exit_outputs, target):
    loss_fn = nn.CrossEntropyLoss()
    loss = 0
    for i, output in enumerate(exit_outputs):
        loss += loss_fn(output, target)
    return loss

class earlyexit_ramp(nn.Module):
    def __init__(self, num_feature=1024, num_classes =10):
        super(earlyexit_ramp, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(num_feature, int(num_feature/2), kernel_size=3, padding=1),
            nn.BatchNorm2d(int(num_feature/2)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(int(num_feature/2), num_classes),
        )
        # self.conv = nn.Conv2d(num_feature, int(num_feature/2), kernel_size=3, padding=1)
        # self.bn = nn.BatchNorm2d(int(num_feature/2))
        # self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(int(num_feature/2), num_classes)
    def forward(self, x):
        return self.head(x)

class EarlyExitHead(nn.Module):
    def __init__(self, in_ch, num_classes):
        super().__init__()
        mid = max(in_ch // 2, 64)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(mid, num_classes),
        )
    def forward(self, x):
        return self.head(x)

class EarlyExitResNet50(nn.Module):
    def __init__(self, original_model, num_class=1000):
        super(EarlyExitResNet50, self).__init__()

        # construct full model list
        detail_layers = []

        # construct early exit list in 
        self.earlyest_point = -15
        self.exit_list = range(self.earlyest_point,-3,1)

        for i, child in enumerate(list(original_model.children())):
            if len(list(child.children())) == 0:
                # print(i, child)
                detail_layers.append(child)
            else: # 
                for j, sub_child in enumerate(list(child.children())):

                    detail_layers.append(sub_child)

        self.features = nn.Sequential(*list(detail_layers[:-15])) # orginal top half model, from -2 to ....
        self.next_layers = nn.ModuleList() # orginal later model layers, 12 items
        self.exit_fcs = nn.ModuleList() # 13 items, not including full model exit

        # self.original_fc = nn.ModuleList()
        # self.original_fc.append(detail_layers[-3])
        # self.original_fc.append(detail_layers[-2])
        # self.original_fc.append(nn.Flatten(start_dim=1))
        # self.original_fc.append(detail_layers[-1])
        self.original_fc = nn.Sequential(detail_layers[-3]
                                         ,detail_layers[-2]
                                         ,nn.Flatten(start_dim=1)
                                         ,detail_layers[-1])
        


        for i in range(self.earlyest_point,-3,1): #[-15,-14,...,-3], totally 12 layers
            self.next_layers.append(detail_layers[i])
        
        self.exit_fcs.append(EarlyExitHead(check_features(self.features), num_class)) # earlyest exit
        for i, tmp_layer in enumerate(self.next_layers): # inject 12 early exit heads after each next_layer, totally 13 exits
            self.exit_fcs.append(EarlyExitHead(check_features(tmp_layer), num_class))
            
        # print('full model exit')
        # for j, child in enumerate(list(self.original_fc.modules())):
        #     print(j,child)
        # Freeze the features part
        for param in self.features.parameters():
            param.requires_grad = False
        for layer in self.next_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.original_fc.parameters():
            param.requires_grad = False
        
        print('middle layers number:', len(self.next_layers))
        print('early exit ramps number:', len(self.exit_fcs))
        
    def forward(self, x, ramp = None):
        x = self.features(x)
        
        if self.training:# training mode
            early_exit_output = []
            early_exit_output.append(self.exit_fcs[0](x))

            for i,layer in enumerate(self.next_layers):
                x = layer(x)
                early_exit_output.append(self.exit_fcs[i+1](x))
            
        else: # inference mode
            if type(ramp)==int and ramp>0: # early exit
                
                # ramp id: 1,2,3,...,12,13
                # if 13, no next_layers executed
                for i in range(13 - ramp): # [-15,-14,-13,...,-2] -> [13,12,11,10,9,...,1] (ramps id)
                    x = self.next_layers[i](x)

                early_exit_output = self.exit_fcs[13-ramp](x)

            elif ramp == None or ramp==0: # full resnet model self.exit_list[-1]
                for i in range(len(self.next_layers)):
                    x = self.next_layers[i](x)
                x = self.original_fc(x)
                early_exit_output = x
        
        # Continue with the remaining layers
        # main_exit_output = self.main_exit(x)
        
        return early_exit_output