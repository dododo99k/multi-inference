import time,copy,os,pickle,argparse,random,math,shutil,queue,tqdm
from torch.multiprocessing import Pool, Manager, current_process
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import torch

# import ee models
from ee_models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Task():
    def __init__(self, tid=-1, arrive_time = -1, start_time = -1, finish_time = -1, warmup = False):
        # self.model_name = model_name
        self.tid = tid
        self.arrive_time = arrive_time
        self.start_time = start_time
        self.finish_time = finish_time
        self.infer_time = -1
        self.queue_time = -1
        self.warmup_task = warmup

    def finalize(self, if_print=True):
        self.infer_time = self.finish_time - self.start_time
        self.queue_time = self.start_time - self.arrive_time
        if if_print: print('infer time: ', self.infer_time, 'queue time: ', self.queue_time)

def generate_poisson_traffic(arrival_rate, duration):
    # generate arrival time first
    arrival_times, t = [], 0
    while t < duration:
        # Generate the time until the next arrival using an exponential distribution
        inter_arrival_time = np.random.exponential(1 / arrival_rate)
        t += inter_arrival_time
        if t < duration:
            arrival_times.append(t)

    return arrival_times

def generate_traffic(intensity, duration):
    # note: here intensity is in ms, and duration is also ms unit
    # first step: generate arrival time first
    arrival_times, t = [], 0
    while t < duration:
        # Generate the time until the next arrival using an exponential distribution
        inter_arrival_time = np.random.exponential(1 / intensity)
        t += inter_arrival_time
        if t < duration:
            arrival_times.append(t)

    # second step: construct traffic in the time domain 
    traffic = np.zeros(int(duration))
    indexs = np.asarray(arrival_times, dtype=int) # indexs could have multiple at the same position
    uniques, counts = np.unique(indexs, return_counts=True)
    traffic[uniques] = counts # assign counts to the position, not 1

    return np.array(traffic, dtype=int)

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

def load_official_model_dataset():

    work_dir = os.getcwd()

    img = torch.rand(3, 64, 64)

    # Step 1: Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    
    # by default, all are resnet50
    model = torch.load(work_dir+'/weights/resnet50_EE.pth')
    model.eval()
    model.train_mode = False

    # Step 2: Initialize the inference transforms
    preprocess = weights.transforms()

    # Step 3: Apply inference preprocessing transforms
    dataset = preprocess(img).unsqueeze(0)

    # Step 4: Use the model and print the predicted category
    # prediction = model(batch).squeeze(0).softmax(0)
    # class_id = prediction.argmax().item()
    # score = prediction[class_id].item()
    # category_name = weights.meta["categories"][class_id]
    # print(f"{category_name}: {100 * score:.1f}%")
    # print('done')

    return model, dataset


def duplicate_model_dataset(model, dataset, num_proc):

    models, datasets = [], []
    for i in range(num_proc):
        mdl = copy.deepcopy(model)
        mdl.to(device)
        mdl.eval()
        mdl.train_mode = False
        models.append(mdl)

        data = copy.deepcopy(dataset)
        data.to(device) # TODO this code does not make the data to GPU, why?
        datasets.append(data)

    return models, datasets

# Persistent worker function: Retrieves a task from the queue and processes it
def worker(execute_queue, finish_queue, model, dataset, ramp_id = None):
    print(f"entered {current_process().name}. waiting for tasks...", flush=True)
    data = dataset.to(device)
    while True:
        # Blocking call, waits until an item is available in the queue
        task = execute_queue.get()  # Blocks indefinitely until a task is available
        # print(f'debug, task: {task.tid}, process: {current_process().name} ')
        if task is None:  # None is the signal to stop the worker
            print(f"{current_process().name} received stop signal.")
            break

        task.start_time = time.perf_counter()

        outputs = model(data, ramp_id)
        # print(f'get results, task: {task.tid}')
        output = outputs[0]
        output.detach().cpu()# .numpy().argmax() # model.eval() is already set, here to device is still needed, TODO XXX
        # print(f"{current_process().name} process finish task: {task.tid}")
        task.finish_time = time.perf_counter()

        task.finalize(if_print=False)

        if not task.warmup_task:
            finish_queue.put(task) # put it into the finish queue for stats and analysis later

def schedule(buffer_queue, execute_queue):
    # this scheduling action happens for all time intervals, if not action needs to be done, then basically skip
    if buffer_queue.empty():
        pass
    else:
        task = buffer_queue.get()
        execute_queue.put(task)


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
    
    argparser.add_argument('-pd','--intensity',
                           type = float,
                           default=0.5,
                           help='poisson density for user inference request number per ms')
    
    argparser.add_argument('-pn','--process_number',
                           type = int,
                           default=4,
                           help='time duration in ms')
      
    argparser.add_argument('-l','--duration',
                           type = int,
                           default=30000,
                           help='time duration in ms')
    
    argparser.add_argument('-pr','--num_process',
                           type = int,
                           default=5,
                           help='time duration in ms')
    # argparser.add_argument('-ml','--model_list',
    #                        type = list,
    #                        default= ['resnet50'] # ,'resnet101',
    #                        )
    
    args = argparser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method('spawn')
    num_proc = args.num_process
    tid = 0
    sim_interval = 1 #ms
    ramp_id = None
    # total_tasks = []

    # model_name = 'resnet50' # or resnet101
    # models, datasets = load_homo_model_dataset(model_name, num_proc) 
    model, dataset = load_official_model_dataset()
    models, datasets = duplicate_model_dataset(model, dataset, num_proc)

    traffic = generate_traffic(args.intensity, args.duration) # here, intensity and duration are both in ms units

    # Create a manager to manage shared data
    with Manager() as manager:
        print('enter in manager')
        # Shared queue managed by the manager
        buffer_queue = manager.Queue() # this queue is to buffer all the incoming tasks, before they are sent to execute_queue, the scheduler mainly schedule this queue
        execute_queue = manager.Queue() # this queue is to send tasks to workers, XXX once task is put, workers will immediately get it to execute XXX
        finish_queue = manager.Queue() # this queue is to collect stats

        # Create a persistent pool of worker processes 
        pool = Pool(processes=num_proc)

        # pool.starmap(run_task, [(model, dataset) for dataset in datasets for model in models])
        
        # Start worker processes to process tasks in the queue
        for i in range(num_proc):
            pool.apply_async(worker, args=(execute_queue, finish_queue, models[i], datasets[i], ramp_id)) # workers see execute_queue, not buffer_queue
        
        warmup_tasks = 100*num_proc
        for i in range(warmup_tasks):
            task = Task(tid=-1, arrive_time=time.perf_counter(), warmup = True)
            execute_queue.put(task)

        # wait for finsihing all warmup tasks
        while not execute_queue.empty():
            continue
        time.sleep(10) # sleep to wait worker process to start

        start_time = time.perf_counter()

        # main loop to receive tasks
        for t in range(args.duration):
            loop_start_time = time.perf_counter()

            for i in range(traffic[t]):

                task = Task(tid=tid, arrive_time=time.perf_counter())
                # print(task.tid, task.arrive_time)
                buffer_queue.put(task) # always put into the buffer queue

                ## XXX main scheduling here XXX #######
                schedule(buffer_queue, execute_queue) # TODO this is a native scheduler, TBD

                tid += 1 # increas task id

            # this control the traffic arrival at real-world time
            elapsed = time.perf_counter() - loop_start_time
            if elapsed>0.001: print('warning, time slot beyond defined unit, time is ', elapsed)
            while elapsed < 0.001:
                elapsed = time.perf_counter() - loop_start_time
            # if elapsed < 0.001: # interval is 1ms
            #     time.sleep(0.001 - elapsed)
            # else:
            #     print('warning, time slot beyond defined unit, time is ', elapsed)
        
        serving_inference_time = time.perf_counter() - start_time
        print('inference runtime usage:', serving_inference_time)
        pass
        # wait for empty execute queue
        while not execute_queue.empty():
            continue

        ###### post processing (XXX this needs to be done here, if the manager() done, then queue are cleared) ######
        results = {}
        print('finish tasks')
        
        while not finish_queue.empty():
            task = finish_queue.get()
            # task = task_list[0]
            results[str(task.tid)] = copy.deepcopy(task)

        # When you're ready to stop the workers, send None into the queue for each worker
        for _ in range(num_proc): execute_queue.put(None)

        # Close and join the pool to wait for workers to finish
        pool.close()
        pool.join()

        print("All tasks processed and workers stopped.")


    pickle.dump(results, open(f'./homomodel_result/results_{ramp_id}_{args.intensity}_{num_proc}.pkl','wb'))

    total_time = time.perf_counter() - start_time
    print('time usage:', total_time)


    ##### plot figure #######
    # results = pickle.load(open('results.pkl','rb'))
    queue_times = [float(r.queue_time) for r in results.values()]
    infer_times = [float(r.infer_time) for r in results.values()]
    # for i in range(999,-1,-1):
    #     if queue_times[i]> 0.05:
    #         break

    pass
    # plt.switch_backend('agg')
    plt.plot(queue_times, label='Queue Times')
    plt.plot(infer_times, label='Inference Times') #;plt.show()
    plt.legend()
    plt.ylim(0,0.02)
    plt.xlabel('Index')
    # plt.yticks(np.arange(0,0.15, 0.01))
    plt.ylabel('Time')
    # fname = f'{args.intensity}_{num_proc}.jpg'
    plt.savefig(f'./homomodel_result/{ramp_id}_{args.intensity}_{num_proc}.jpg')
    plt.close()
    # plt.show()