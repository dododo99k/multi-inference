import time,copy,os,pickle,argparse,random,math,shutil,queue,tqdm,json,gc
from collections import deque
from torch.multiprocessing import Pool,Process, Semaphore, Event, Manager, Value, Queue, current_process
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from functions import Task, FastRollingWindow, InferenceRecord
from ee_models import *
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights, vgg11, VGG11_Weights, vgg19, VGG19_Weights
from scheduler import Scheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_official_models_datasets():

    models = []
    model = torch.load('./weights/resnet50_EE.pth',map_location='cpu', weights_only=False)
    model.eval()
    # model.to(device)
    models.append(model)

    datasets = []
    datasets.append(torch.randn(1, 3, 224, 224)) 

    # for dataset in datasets:
    #     dataset.to(device)

    return models, datasets

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-s','--seed',
                           type=int,
                           default=43,
                           help='random seed')
    
    argparser.add_argument('-pd','--intensity',
                           type = float,
                           default=0.5,
                           help='poisson density for user inference request number per ms')
      
    argparser.add_argument('-l','--duration',
                           type = int,
                           default=8,
                           help='time duration in seconds')
         
    argparser.add_argument('-b','--batch',
                           type = bool,
                           default=True,
                           help='if using batch workers')
        
    args = argparser.parse_args()
    unit = 1000 # ms
    args.duration = args.duration * unit # convert to ms unit
    sim_interval = 1/unit #ms
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method('spawn')

    # model_names = ['resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg19']
    model_name = 'resnet50ee'
    # load models and weights and datasets from PyTorch official archive
    models, datasets = load_official_models_datasets()
    
    num_proc = len(models)
    
    model = models[0].to(device).eval()

    data = torch.randn(50, 3, 224, 224).to(device)
    with torch.inference_mode():
        # It is more memory-efficient to generate data on-the-fly for warm-up
        # instead of pre-allocating and holding a large list of tensors in GPU memory.
        # The high memory usage is an artifact of PyTorch's caching allocator.
        # When warming up with increasing batch sizes, the allocator can reserve far
        # more memory than is strictly necessary due to fragmentation.
        # To mitigate this, we warm up from the largest batch size downwards.
        # This allows the allocator to reuse large memory blocks for smaller batches.
        for _ in range(1):
            for datasize in range(1, 51): # Iterate from 49 down to 1
                for ee in range(0, 14):
                    _ = model(data[:datasize], ee)
    print(f"batch {current_process().name} warmed up.", flush=True)

    #### measure time ####
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        # Generate and process data on-the-fly to minimize memory usage
        for datasize in range(1, 51):
            for ee in range(0, 14):
                _ = model(data[:datasize], ee)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"batch {current_process().name} inference time: {end_time - start_time:.2f} seconds.", flush=True)
    
    # Example of how to clear GPU memory
    print("\n--- Demonstrating GPU memory clearing ---")
    if device.type == 'cuda':
        print(f"Memory allocated before clearing: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached before clearing: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

        # Remove references to model and data
        # del model
        # del random_data_list # This list no longer exists with the new approach
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\nMemory allocated after clearing: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory cached after clearing: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

    #### measure time again after deleting and clearing cache ####
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.inference_mode():
        # Generate and process data on-the-fly to minimize memory usage
        for datasize in range(1, 51):
            for ee in range(0, 14):
                _ = model(data[:datasize], ee)
                # torch.cuda.empty_cache()
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"batch {current_process().name} inference time: {end_time - start_time:.2f} seconds.", flush=True)
    pass