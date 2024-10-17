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
        self.conv = nn.Conv2d(num_feature, int(num_feature/2), kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(int(num_feature/2))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(num_feature/2), num_classes)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
class EarlyExitResNet50(nn.Module):
    def __init__(self, original_model, num_class=1000):
        super(EarlyExitResNet50, self).__init__()

        # construct full model list
        self.train_mode = True
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
                    # print(i,j, sub_child)
                    # print('***********************************')
                    detail_layers.append(sub_child)

        self.features = nn.Sequential(*list(detail_layers[:-15])) # orginal top half model, from -2 to ....
        self.next_layers = nn.ModuleList() # orginal later model layers, 12 items
        self.exit_fcs = nn.ModuleList() # 13 items, not including full model exit

        self.original_fc = nn.ModuleList()
        self.original_fc.append(detail_layers[-3])
        self.original_fc.append(detail_layers[-2])
        self.original_fc.append(nn.Flatten(start_dim=1))
        self.original_fc.append(detail_layers[-1])
            
        print('features output:', (check_features(self.features)))
        self.exit_fcs.append(earlyexit_ramp(check_features(self.features), num_class)) # earlyest exit


        for i in range(-15,-3,1): #[-15,-14,...,-3] 13 ramps
            # tmp_layer = nn.Sequential(*list(detail_layers[i].children()))
            tmp_layer = detail_layers[i]
            self.next_layers.append(tmp_layer)
            print(i,' next layers output:', (check_features(tmp_layer)))
            for j, child in enumerate(list(tmp_layer.children())):
                print(j,child)
            print()
            self.exit_fcs.append(earlyexit_ramp(check_features(tmp_layer), num_class))
        
        print('full model exit')
        for j, child in enumerate(list(self.original_fc.modules())):
            print(j,child)
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
        early_exit_output = []
        x = self.features(x) # 
        if self.train_mode:
            early_exit_output.append(self.exit_fcs[0](x))

            for i,layer in enumerate(self.next_layers):
                x = layer(x)
                early_exit_output.append(self.exit_fcs[i+1](x))
            # early_exit_output.append(self.exit_fcs[i+1](x))
        else: # inference mode
            if type(ramp)==int: # early exit
                # for layer in self.next_layers:
                for i in range(ramp): # [-15,-14,-13,...,-2] -> [0,1,2,3,4,...] (ramps id)
                    x = self.next_layers[i](x)

                early_exit_output.append(self.exit_fcs[ramp](x))

            elif ramp == None: # full resnet model self.exit_list[-1]
                for i in range(len(self.next_layers)):
                    x = self.next_layers[i](x)
                for i in range(len(self.original_fc)):
                    x = self.original_fc[i](x)
                    # print(x.size())

                early_exit_output.append(x)
        
        # Continue with the remaining layers
        # main_exit_output = self.main_exit(x)
        
        return early_exit_output# , main_exit_output
