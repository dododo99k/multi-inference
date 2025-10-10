import torch
import torchvision
import time
import os
import pickle
import torchvision.models as models
from PIL import Image
import copy

device = torch.device('cuda')

# 装饰器，用于计算推理耗时
def getCompDuration(func):
    def wrapper(*args, **kwargs):
        print("%s is running" % func.__name__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        computeTime = end - start
        print('forward time cost: %.5f sec' %computeTime)
        return result
    return wrapper

def getInputData(device = device):
    # weights = models.ResNet152_Weights.DEFAULT
    weights = models.ResNet152_Weights.IMAGENET1K_V2
    preprocess = weights.transforms()
    img = Image.open('test.png').convert("RGB")
    img = preprocess(img).unsqueeze(0).to('cuda')
    return img

# @getCompDuration
def modelForwardImage(input, model, nTimes, nProcess, stream, device = device):
    pid = os.getpid()
    model.eval()
    latency_list = []
    for i in range(nTimes):
        
        start = time.perf_counter()
        with torch.no_grad(): # 非常重要，用于降低显存消耗
            predictions = model(input)
        end = time.perf_counter()

        computeTime = end - start
        latency_list.append(computeTime)
    pickle.dump(latency_list, open('./'+str(nProcess)+'/'+str(pid)+'.pkl', 'wb'))
            # pred = predictions[0]['boxes'].shape
            # print(f'pid:{pid}, stream:{stream}')

def getModels(nProcess, device = device):
    modellist = []
    with torch.no_grad():
        weights = models.ResNet152_Weights.IMAGENET1K_V2
        model = models.resnet152(weights=weights).eval().to('cuda')
        for i in range(nProcess):
            mod = copy.deepcopy(model)
            modellist.append(mod)
        return modellist
    
def funcInStream(input, model, nTimes, nProcess):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        modelForwardImage(input, model, nTimes, nProcess, stream)
        
def test(nTimes, nProcess):
    input = getInputData()
    
    # spwan是多任务的一种方式
    ctx = torch.multiprocessing.get_context('spawn')
   
    models = getModels(nProcess)
    pool = []
    for i in range(nProcess):
        p = ctx.Process(target = funcInStream, args = (input, models[i], nTimes, nProcess))
        pool.append(p)
    
    for p in pool:
        p.start()

if __name__ == '__main__':
    nTimes = 10000
    nProcess = 4

    path = os.getcwd() + '/' + str(nProcess)
    if not os.path.isdir(path):
        os.mkdir(path)
    
    test(nTimes, nProcess)
