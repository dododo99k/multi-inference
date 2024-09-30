import time,copy,os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from ee_models import EarlyExitResNet50 # do not delete this line
import matplotlib.pyplot as plt

iteration_num = 1

# work_dir = os.path.dirname(os.getcwd())
work_dir = os.getcwd()

# define data tansform function
weights = models.ResNet101_Weights.DEFAULT
transform = weights.transforms()

testset = torchvision.datasets.CIFAR10(root= work_dir+'/dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
model = torch.load(work_dir+'/weights/resnet101.pth')
# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.to(device)

model.eval()

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
            outputs = model(images)
            inference_time = time.perf_counter() - start_time
            total_time += inference_time
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            break

print(f'original model, inference time usage: {total_time/iteration_num}, accuracy: {100 * correct / total} %')
