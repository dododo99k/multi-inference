# standard packages
import time,copy,os,pickle,argparse,random,math,shutil,queue,tqdm
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
# torch 
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
from torchvision import models

# my models defination
# since torch use pickle to save model structure/parameters
from ee_models import *

model_list = [
    'resnet50','resnet50_EE',
    'resnet101','resnet101_EE'
]


class inference_request():
    def __init__(self, model_index = 0, arrive_server_time = 0,start_time = 0, finish_time = 0):
        self.model_index = model_index
        self.arrive_server_time = arrive_server_time
        self.start_time = start_time
        self.finish_time = finish_time
    # def update_model(self, model_index):
    #     self.model_index = model_index
    # def update_arrive_time(self, arrive_server_time):
    #     self.arrive_server_time = arrive_server_time
    # def update_start_time(self, start_time):
    #     self.start_time = start_time
    # def update_finish_time(self, finish_time):
    #     self.finish_time = finish_time


class env_main():
    def __init__(self, seed = 4096,
                user_number = 1,
                poisson_density = 0.01,
                length = 1000,
                model_list = ['resnet50']):
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.user_number = user_number
        self.time = 0
        self.time_length = length

        self.active_workers = [] # equal to cuda_stream number
        self.active_workers_num = 0
        self.max_workers = 10


        # load models
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_list = model_list
        self.models = []
        self.ee_models =[]
        for model_name in model_list:
            model = torch.load('./weights/'+ model_name+'.pth')
            model.to(device)
            model.eval()
            self.models.append(model)

            # self.ee_models.append(torch.load('./weights/'+ model+'_EE.pth'))

        for m in self.models:
            print(m.__class__.__name__)
            # print(m.DEFAULT)

        # inference queue
        self.inference_wait_queue = queue.Queue()
        self.inference_finish_queue = mp.Queue()
        # data loader
        work_dir = os.getcwd()
        transform = models.ResNet101_Weights.DEFAULT.transforms()
        testset = torchvision.datasets.CIFAR10(root= work_dir+'/dataset', train=False, download=True, transform=transform)
        resnet50_data_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
        # dataiter = iter(resnet50_data_loader)
        resnet50_images, resnet50_labels = next(iter(resnet50_data_loader))
        self.resnet50_images, self.resnet50_labels = resnet50_images.to(device), resnet50_labels.to(device)
              
        transform = models.ResNet101_Weights.DEFAULT.transforms()
        testset = torchvision.datasets.CIFAR10(root= work_dir+'/dataset', train=False, download=True, transform=transform)
        resnet101_data_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
        # dataiter = iter(resnet101_data_loader)
        resnet101_images, resnet101_labels = next(iter(resnet101_data_loader))
        self.resnet101_images, self.resnet101_labels = resnet101_images.to(device), resnet101_labels.to(device)

        transform = models.MobileNet_V2_Weights.DEFAULT.transforms()
        testset = torchvision.datasets.CIFAR10(root= work_dir+'/dataset', train=False, download=True, transform=transform)
        mobilenet_data_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)
        # dataiter = iter(mobilenet_data_loader)
        mobilenet_images, mobilenet_labels = next(iter(mobilenet_data_loader))
        self.mobilenet_images, self.mobilenet_labels = mobilenet_images.to(device), mobilenet_labels.to(device)
        #######################  ego vehicle ppp look up table (pdf), for fast ppp sample
        ppp_CDF = []
        cdf = 0
        self.ppp_lambda = poisson_density # 0.02 by deault
        n = 0 
        while True:
            prob = self.ppp_lambda**n/math.factorial(n)*math.exp(-self.ppp_lambda)
            cdf += prob
            ppp_CDF.append(cdf)
            # print(n, pdf)
            if cdf >= 1 or ( n > self.ppp_lambda and prob <= 1e-16): # float number precision
                break
            n+=1
        self.ppp_CDF = ppp_CDF
    
    def inference_request_generator(self):
        # return two values
        # mid is the current time frame inference requests number
        # model_id is the index of model of self.model_list
        sample_pdf = random.random()

        left, right = 0, len(self.ppp_CDF) - 1
        while left < right:
            mid = left + (right - left) // 2
            if self.ppp_CDF[mid] < sample_pdf:
                left = mid + 1
            elif self.ppp_CDF[mid] > sample_pdf:
                right = mid - 1
        mid = left + (right - left) // 2

        if self.ppp_CDF[mid] < sample_pdf and sample_pdf <= self.ppp_CDF[mid+1]:
            mid+=1
        elif self.ppp_CDF[mid+1] < sample_pdf:
            mid+=2
        
        return mid
    
    def choose_model_index(self):
        return random.randint(0, len(self.model_list)-1)
    
    def data_loader(self, model_index):
        # different model has different data loader
        # use modelname to speficy weight transform
        model_name = self.model_list[model_index]
        if 'resnet50' in model_name:
            return self.resnet50_images, self.resnet50_labels
        elif 'resnet101' in model_name:
            return self.resnet101_images, self.resnet101_labels
        elif 'resnet101' in model_name:
            return self.mobilenet_images, self.mobilenet_labels

    
    def add_worker(self, request):
        # create a new cuda stream
        request.start_time = time.time()
        input_data, label = self.data_loader(request.model_index)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            with torch.no_grad():
                result = self.models[model_index](input_data)
        result = result.cpu()
        request.finish_time = time.time()
        self.inference_finish_queue.put(request)

    def step(self):
        Infer_num = self.inference_request_generator() # number of user inference request at current time frame
        for i in range(Infer_num):
            model_index = self.choose_model_index()
            
            request = inference_request(model_index, time.time(), 0, 0)

            self.inference_wait_queue.put(request)
            pass
    
    def schdule(self):
        wait_queue_size = self.inference_wait_queue.qsize()

        # direct issue request
        for _ in range(wait_queue_size):
            request = self.inference_wait_queue.get()
            p = mp.Process(target=self.add_worker, args=(request,))
            p.start()
            self.active_workers.append(p)
        
        for p in self.active_workers:
            if not p.is_alive():
                print('process finish')
                self.active_workers.remove(p)
        self.active_workers_num = len(self.active_workers)

        # for i in range(self.active_workers_num):
        #     if not self.active_workers[i].is_alive():
        #         print('process finish')
        #         del_list.append(i)


        # pass
# use another process to mananger all inferences


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-s','--seed',
                           type=int,
                           default=4096,
                           help='random seed')
    
    argparser.add_argument('-un','--user_number',
                           type = int,
                           default=1,
                           help='user number')
    
    argparser.add_argument('-pd','--poisson_density',
                           type = float,
                           default=0.02,
                           help='poisson density for user inference request number')
    
    argparser.add_argument('-l','--length',
                           type = int,
                           default=1000,
                           help='time length in ms')
    argparser.add_argument('-ml','--model_list',
                           type = list,
                           default= ['resnet50','resnet50_EE','resnet101','resnet101_EE']
                           )
    args = argparser.parse_args()

    env = env_main(seed = args.seed,
                   user_number = args.user_number,
                   poisson_density = args.poisson_density,
                   length = args.length,
                   model_list = args.model_list
                   )
    # warm up
    # TODO
    start_time = time.time()
    time_length_s = args.length/1000
    # real time
    # schedule_process = mp.Process(target=schedule, args=(task_queue, result_queue, num_gpus, max_processes))
    # schedule_process.start()
    while True:
        loop_start_time = time.time()
        current_time = loop_start_time - start_time
        if current_time >= time_length_s:
            break
        # generate user inference request
        num = env.inference_request_generator()
        # random choose model
        for i in range(num):
            model_index = env.choose_model_index()
            request = inference_request(model_index, time.time(), 0, 0)
            env.inference_wait_queue.put(request)
            env.schdule()
        elapsed = time.time() - loop_start_time
        if elapsed < 0.001:
            time.sleep(0.001 - elapsed)
        else:
            print('warning, time slot beyond defined unit')
    total_time = time.time() - start_time

    # wait for all process fin
    while True:
        loop_start_time = time.time()
        if env.active_workers_num == 0:
            break
        elapsed = time.time() - loop_start_time
        if elapsed < 0.001:
            time.sleep(0.001 - elapsed)
        else:
            print('warning, time slot beyond defined unit')
    print('time usage:',total_time)
    