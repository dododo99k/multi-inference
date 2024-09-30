import time,copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from ee_models import EarlyExitResNet50 # do not delete this line

iteration_num = 50

# define data tansform function
weights = models.ResNet50_Weights.DEFAULT
transform = weights.transforms()

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=16)
model = torch.load('./weights/resnet50.pth')

ee_model = torch.load('./weights/resnet50_EE.pth')

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.to(device)
ee_model.to(device)

model.eval()
ee_model.eval()

# warm up gpu
for _ in range(10):
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            start_time = time.perf_counter()
            _ = model(images)
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
            outputs = model(images)
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