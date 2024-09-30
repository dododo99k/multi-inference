import time
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torchsummary import summary

iterations = 10000

weights = models.ResNet50_Weights.DEFAULT
print(weights)
resnet = models.resnet50(weights=weights).eval().to('cuda')

# summary(resnet, (3, 224, 224))

preprocess = weights.transforms()

# img = Image.open('test.png').convert("RGB")
# img = preprocess(img).unsqueeze(0).to('cuda')


# start_time = time.perf_counter()

# for i in range(iterations):
#     with torch.no_grad():
#         output = resnet(img)

# time_usage = time.perf_counter()-start_time
# print(f'total time: {time_usage}')