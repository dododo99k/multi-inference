import time,copy,os,pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.io import read_image
from ee_models import EarlyExitResNet50 # do not delete this line
import matplotlib.pyplot as plt

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

iteration_num = 1

# work_dir = os.path.dirname(os.getcwd())
work_dir = os.getcwd()

# define data tansform function
weights = models.ResNet50_Weights.DEFAULT
transform = weights.transforms()

# testset = torchvision.datasets.CIFAR10(root= work_dir+'/dataset', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=16)
torch.manual_seed(1)
data = torch.randn(1, 3, 224, 224).to(device) 
# model = torch.load(work_dir+'/weights/resnet50.pth')
model = models.resnet50(weights=None)
model.to(device)
model.eval()

# model
total_time = 0
time_record = []

with torch.no_grad(): 
    for _ in range(1):
        start_time = time.perf_counter()
        outputs = model(data)
        outputs.detach().cpu()
        end_time = time.perf_counter()
        time_record.append(end_time-start_time)


plt.plot(time_record, label='Inference Times') #;plt.show()
plt.legend()
plt.ylim(0,0.02)
# plt.savefig('./homomodel_result/2'+str(time.time())+'.jpg')
plt.show()
plt.close()

# for _ in range(iteration_num):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for counter, data in enumerate(testloader):
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
#             start_time = time.perf_counter()
#             outputs = model(images)
#             inference_time = time.perf_counter() - start_time
#             time_record.append(inference_time)
#             total_time += inference_time
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#             break

# print(f'original model, inference time usage: {total_time/iteration_num}, accuracy: {100 * correct / total} %')
# print(f'average time: {np.mean(time_record)}, time variance: {np.var(time_record)}')
# print(f'ignoring warm up time, average time: {np.mean(time_record[1:])}, time variance: {np.var(time_record[1:])}')

# pickle.dump(time_record, open('./resnet50 inference time.pkl', 'wb'))
