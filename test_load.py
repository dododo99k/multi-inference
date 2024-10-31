import time,copy,os,pickle, argparse
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-s','--seed',
                           type=int,
                           default=42,
                           help='random seed')
    
    argparser.add_argument('-l','--length',
                           type = int,
                           default=1000,
                           help='num of iterations')

    args = argparser.parse_args()

    #### model ##################
    torch.manual_seed(args.seed)
    mods = []

    mods.append(models.resnet50(weights=models.ResNet50_Weights.DEFAULT))
    mods.append(models.resnet101(weights=models.ResNet101_Weights.DEFAULT))
    mods.append(models.resnet152(weights=models.ResNet152_Weights.DEFAULT))
    mods.append(models.vgg11(weights=models.VGG11_Weights.DEFAULT))
    mods.append(models.vgg16(weights=models.VGG16_Weights.DEFAULT))
    mods.append(models.vgg19(weights=models.VGG19_Weights.DEFAULT))
    
    for mod in mods: 
        mod.to(device)
        mod.eval()

    #### test data ##############
    data = torch.randn(1, 3, 224, 224).to(device) 

    # model
    total_time = 0
    time_record = []

    with torch.no_grad(): 
        for _ in range(10):
            for mod in mods:
                start_time = time.perf_counter()
                outputs = mod(data)
                outputs.detach().cpu()
                end_time = time.perf_counter()
                time_record.append(end_time-start_time)

    print('resnet50', time_record[-6], 
          'resnet101', time_record[-5], 
          'resnet152', time_record[-4], 
          'vgg11', time_record[-3],
          'vgg16', time_record[-2],
          'vgg19', time_record[-1])

