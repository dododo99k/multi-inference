import time
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

iterations = 10000

weights = models.ResNet152_Weights.DEFAULT
preprocess = weights.transforms()

model = models.resnet152(weights=weights).eval().to('cuda')



img = Image.open('test.png').convert("RGB")
img = preprocess(img).unsqueeze(0).to('cuda')


start_time = time.perf_counter()

for i in range(iterations):
    with torch.no_grad():
        output = model(img)

time_usage = time.perf_counter()-start_time
print(f'total time: {time_usage}')