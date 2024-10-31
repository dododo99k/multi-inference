import time,copy,os,pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
from ee_models import EarlyExitResNet50 # do not delete this line
import matplotlib.pyplot as plt

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


torch.manual_seed(1)
data = torch.randn(1, 3, 224, 224).to(device) 
# model = torch.load(work_dir+'/weights/resnet50.pth')
model = resnet50(weights=ResNet50_Weights)
model.to(device)
model.eval()

# model
total_time = 0
infer_times = []

 
for _ in range(3000):
    # with torch.no_grad():
    start_time = time.time()
    outputs = model(data)
    outputs.detach().cpu()
    end_time = time.time()
    infer_times.append(end_time-start_time)


print(np.mean(infer_times), np.std(infer_times))
# plt.switch_backend('agg')
plt.plot(infer_times, label='Inference Times') 
plt.legend()
plt.ylim(0,0.02)
plt.xlabel('Index')
# plt.yticks(np.arange(0,0.15, 0.01))
plt.ylabel('Time')
# fname = f'{args.intensity}_{num_proc}.jpg'
plt.savefig('./test_run_proccess'+str(2)+'_'+str(int(np.mean(infer_times)*10000))+'_'+str(int(np.std(infer_times)*10000))+'.jpg')
plt.close()
