import time,copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models

iteration_num = 50

# define data tansform function
weights = models.MobileNet_V2_Weights.DEFAULT
transform = weights.transforms()

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=16)

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define mobilenetV2_cifar10
base_model = models.mobilenet_v2(weights=weights)
base_model.classifier[1] = nn.Linear(base_model.last_channel, 10)  # CIFAR-10 has 10 classes
base_model.load_state_dict(torch.load('./weights/MobileNetV2.pth'))

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

num_classes = 10  # 根据你的任务更改类别数量
exit_layer = -6
ee_model = EarlyExitMobileNet(base_model, exit_layer , num_classes)


# load the original weights
base_model.load_state_dict(torch.load('./weights/mobilenetv2.pth'))
ee_model.load_state_dict(torch.load('./weights/EE'+str(exit_layer)+'_MobileNetV2.pth'))

base_model.to(device)
ee_model.to(device)

base_model.eval()
ee_model.eval()


# warm up gpu
for _ in range(10):
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            _ = base_model(images)
            inference_time = time.perf_counter() - start_time
# model

total_time = 0

for _ in range(iteration_num):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = base_model(images)
            inference_time = time.perf_counter() - start_time
            total_time += inference_time
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(f'original model, inference time usage: {total_time/iteration_num}, accuracy: {100 * correct / total} %')


# ee_model
total_time = 0
for _ in range(iteration_num):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            outputs = ee_model(images)
            inference_time = time.perf_counter() - start_time
            total_time += inference_time
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(f'Early exit model, inference time usage: {total_time/iteration_num}: {100 * correct / total} %')
