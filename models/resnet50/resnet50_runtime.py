import time,copy
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from resnet50_ee_train import earlyexit_ramp, EarlyExitResNet50 # do not delete this line

iteration_num = 5

# define data tansform function
weights = models.ResNet50_Weights.DEFAULT
transform = weights.transforms()

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=transform)
testset = torch.utils.data.Subset(testset, range(100))
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)
# model = torch.load('./weights/resnet50.pth', weights_only=False)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
ee_model = torch.load('./weights/resnet50_EE.pth', weights_only=False)

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model.eval()
ee_model.eval()

model.to(device)
ee_model.to(device)



# warm up gpu
print("Warming up GPU..., model")
for _ in range(5):
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            _ = model(images)

# model
total_time = 0
counter = 0
print("Testing inference time and accuracy...")
for _ in range(iteration_num):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = model(images)
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start_time
            counter += 1
            total_time += inference_time
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(f'original model, inference time usage: {total_time/counter*1000} ms, accuracy: {100 * correct / total} %')

time.sleep(5)
print("Warming up GPU..., ee model")
for _ in range(5):
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            _ = ee_model(images)
print("Testing EE Model inference time and accuracy...")
# ee_model
total_time = 0
counter = 0
for _ in range(iteration_num):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = ee_model(images)
            torch.cuda.synchronize()
            inference_time = time.perf_counter() - start_time
            total_time += inference_time
            counter += 1
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

print(f'Early exit model, inference time usage: {total_time/counter*1000} ms, accuracy: {100 * correct / total} %')