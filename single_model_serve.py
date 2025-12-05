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



def load_official_models_datasets():

    models = []
    ee_model = torch.load('./weights/resnet50_EE.pth',map_location='cpu', weights_only=False)
    ee_model.eval()
    # ee_model.to(device)
    models.append(ee_model)

    datasets = []
    datasets.append(torch.randn(1, 3, 224, 224)) 

    # for dataset in datasets:
    #     dataset.to(device)

    return models, datasets


# Persistent worker function: Retrieves a task from the queue and processes it
def batch_worker(model, execute_queue, permit, done, stop, shared_mem_list, data_number, ee_head):
    # permit: Semaphore to control access to the shared memory
    # done: Event to signal that processing is done
    # stop: Event to signal that processing should stop
    # shared_mem_list: shared memory tensor to read data from
    # data_number: Value to indicate the number of data items in shared memory
    os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
    # Robust device selection; fallback to CPU if CUDA is unavailable/problematic
    try:
        torch.cuda.set_device(0)
        dev = torch.device('cuda:0')
    except Exception:
        dev = torch.device('cpu')
    print(f'{current_process().name} using device {dev}')
    model = model.to(dev).eval()
    random_data = torch.randn(50, 3, 224, 224).to(dev)
    
    # warm up
    with torch.inference_mode():
        for datasize in reversed(range(1, 51)):
            for ee in range(0, 14):
                _ = model(random_data[:datasize],ee)
                
    # for ee in range(0, 14):
    #     _ = model(random_data, ee)
    print(f"[Worker] batch {current_process().name} warmed up.", flush=True)
    
    
    
    # serving loop
    while True:
        permit.wait()           # 阻塞等待许可（不占 CPU）
        if stop.is_set():          # 允许优雅收尾
            print(f"[Worker] {current_process().name} received stop signal. Exiting.")
            break
        receive_time_temp = time.time_ns()
        # print(f"{current_process().name}, shared memory pointer {data_number.value}, ee_head {ee_head.value}")
        torch.cuda.synchronize()  # 确保数据已传输到 GPU
        # data = shared_mem_list[:data_number.value].to(dev)
        if execute_queue.qsize() != data_number.value:
            print(f"[WARNING] {current_process().name} execute_queue size {execute_queue.qsize()} not equal to data_number {data_number.value}")
            
        data = random_data[:data_number.value]
        
        start_time_temp = time.time_ns()
        _ee_head = int(ee_head.value)
        # print(f'[Worker debug] start inference batch size {data.size(0)}, ee_head \033[31m{_ee_head}\033[0m')
        results = model(data, _ee_head)  # 批处理推理
        
        torch.cuda.synchronize()  # 确保推理完成
        finish_time_temp = time.time_ns()
        # results.detach().cpu()     # 释放许可，通知主进程批处理已完成
        
        tmp = []
        while True:
            try:
                task = execute_queue.get_nowait()
            except queue.Empty:
                break
            task.ee_head = _ee_head
            task.start_time = (start_time_temp - task.base_time) / 1_000_000
            task.finish_time = (finish_time_temp - task.base_time) / 1_000_000
            tmp.append(task)
        for task in tmp:
            execute_queue.put(task)
        
        permit.clear()            # 重置许可,等待下一批数据
        done.set()

        # Disable print to avoid I/O blocking that causes ~70ms delay
        # print(f"{current_process().name} finished processing batch of size {len(tmp)}, time: {(time.time_ns() - receive_time_temp) / 1_000_000} ms, inference time: {(finish_time_temp - start_time_temp) / 1_000_000} ms.")




#### TODO XXX
#### increase the arrival interval, does not have to be 1ms granularity

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
    
    
    # total_tasks = []

    # generate traffic 
    traffic = generate_traffic(args.intensity, args.duration) # here, intensity and duration are both in ms units
    print('generated totally', sum(traffic), 'tasks')

    # Create a manager to manage shared data


    # Shared queue (per model) managed by the manager
    buffer_queues = [deque() for _ in range(num_proc)] # this queue is to buffer all the incoming tasks, before they are sent to execute_queue, the scheduler mainly schedule this queue
    execute_queues = [Queue() for _ in range(num_proc)] # this multiprocessing queue is to send tasks to workers, XXX once task is put, workers will immediately get it to execute XXX
    finish_queues = [deque() for _ in range(num_proc)] # this queue is to collect stats

    shared_mem_lists = [torch.empty(50, 3, 224, 224).share_memory_() for _ in range(num_proc)]
    shared_mem_indexs = [Value('i', 0) for _ in range(num_proc)] # indicate the current index to write new data
    # permits = [Semaphore(1) for _ in range(num_proc)]
    permits = [Event() for _ in range(num_proc)]  
    dones = [Event() for _ in range(num_proc)]
    stop = Event()
    ee_heads = [Value('i', -1) for _ in range(num_proc)] # indicate which ee head to use for each model

    with open(f'./profile_results/{model_name}_inference_times.json', "r", encoding="utf-8") as f:
        model_profile = np.array(json.load(f))
        
    scheduler = Scheduler(num_proc, buffer_queues[0], execute_queues[0], finish_queues[0], shared_mem_lists[0], shared_mem_indexs[0],\
        permits[0], dones[0], stop, model_profile=model_profile, ee_head=ee_heads[0])
    # finish_times = [Value('d', 0.0) for _ in range(num_proc)] # store finish time of each batch worker
    
    
    # Start worker processes to process tasks in the queue
    torch_processes = []
    for i in range(num_proc):
        p = Process(
                target=batch_worker,
                args=(models[i], execute_queues[i], permits[i], dones[i], stop, shared_mem_lists[i], shared_mem_indexs[i], ee_heads[i]),
                name=f'Worker-{i}'
            )
        p.start()
        torch_processes.append(p)
    # wait for workers to start
    print('waiting for workers to start and warm up...')
    time.sleep(10)

    # start_time = time.perf_counter()
    print('starting the main loop...')
    
    #########################
    # generate tasks ahead of simulation
    #########################
    tkid = 0
    tasks_list = [] # store tasks at each time slot, length should be args.duration
    task_num_per_model_lists = [] # store how many tasks for each model at each time slot
    for t in range(args.duration):
        temp = []
        task_num_per_model_temp = [0]*num_proc
        for i in range(traffic[t]):
            # idx = np.random.randint(num_proc)
            idx = 0 # single model serving
            task_num_per_model_temp[idx] += 1
            task = Task(model_id=idx, tk_id=tkid, slo=25)
            tkid += 1 # increas task id
            temp.append(task)
        tasks_list.append(temp)
        task_num_per_model_lists.append(task_num_per_model_temp)
    #########################
    # main loop to receive tasks
    #########################
    base_time = time.time_ns()
    loop_base_time = time.perf_counter()
    for t in range(args.duration):
        # print(f'[SIM TIME] {t} ms')        
        task_num_per_model = task_num_per_model_lists[t] # how many tasks for each model at current time slot
        # put tasks into buffer queues
        _arrive_time = (time.time_ns()-base_time) / 1_000_000
        for task in tasks_list[t]:
            task.base_time = base_time
            task.arrive_time = _arrive_time
            model_id = task.model_id
            input_data = task.input_data
            # put data into shared memory
            # print(f'[DEBUG] shared memory index {shared_mem_indexs[model_id].value}')
            # shared_mem_lists[model_id][shared_mem_indexs[model_id].value].copy_(input_data.squeeze(0))
            # shared_mem_indexs[model_id].value += 1
            buffer_queues[model_id].append(task)
        
        # finish tasks from GPU
        scheduler.finish_tasks_from_gpu()
        # scheduling
        scheduler.schedule()

        elapsed = time.perf_counter() - loop_base_time
        if elapsed < (t+1)/unit: # interval is 1 ms = 0.001 s
            time.sleep((t+1)/unit - elapsed)
        else:
            print(f'[WARNING!!!!!!!], {t} ms')
            
    for t in range(1000): # extra 1000 ms to process remaining tasks
        loop_start_time = time.perf_counter()
        scheduler.finish_tasks_from_gpu()
        scheduler.schedule()
        # this control the traffic arrival at real-world time
        elapsed = time.perf_counter() - loop_start_time
        if elapsed < sim_interval: # interval is 1 ms = 0.001 s
            time.sleep(sim_interval - elapsed)
        else:
            print('.', end='', flush=True)

    scheduler.stop_all()
    # wait for tasks to complete
    print('wait for 5 seconds...')
    time.sleep(5)
    # wait for empty execute queue
    while True:
        empty_all = True
        for execute_queue in execute_queues:
            print(f'execute queue size: {execute_queue.qsize()}', end='; ')
            empty_all = empty_all and execute_queue.empty()
        if empty_all: break
        print('wait for 1 more second...')
        time.sleep(1)
        
    
    ###### post processing (XXX this needs to be done here, if the manager() done, then queue are cleared) ######
    all_results = {}
    total_time_list = []
    queue_time_list = []
    infer_time_list = []
    ee_heads_list = []
    for proc in range(num_proc):
        results = {}
        while len(finish_queues[proc]) > 0:
            task = finish_queues[proc].popleft()
            
            total_time_list.append(task.total_time)
            queue_time_list.append(task.queue_time)
            infer_time_list.append(task.infer_time)
            ee_heads_list.append(task.ee_head)
            # task = task_list[0]
            results[str(task.tk_id)] = copy.deepcopy(task)
        all_results[model_name] = results
            
    # print('collected totally', len(results.keys()), 'tasks')

    # When you're ready to stop the workers, send None into the queue for each worker
    for proc in range(num_proc): 
        execute_queues[proc].put(None)

    # Close and join the pool to wait for workers to finish

    print("All tasks processed and workers stopped.")

    os.makedirs(os.path.dirname('./single_model_result/'), exist_ok=True)
    pickle.dump(all_results, open(f'./single_model_result/results_intensity_{args.intensity}.pkl','wb'))

    print('results are saved')
    print('total tasks number in scheduler buffer history:', sum(scheduler.buffer_history))
    # draw total time CDF and total time series
    sorted_total_time = np.sort(np.array(total_time_list))
    sorted_queue_time = np.sort(np.array(queue_time_list))
    sorted_infer_time = np.sort(np.array(infer_time_list))
    sorted_ee_heads = np.sort(np.array(ee_heads_list))
    plt.figure(figsize=(18,10))
    outer_gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    top_gs = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer_gs[0], wspace=0.25)
    bottom_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_gs[1], width_ratios=[1, 2], wspace=0.3)
    
    ax0 = plt.subplot(top_gs[0])
    ax0.plot(sorted_total_time, np.arange(len(sorted_total_time))/len(sorted_total_time), marker='o', markersize=2)
    ax0.set_xlabel('Total Time (ms)')
    ax0.set_ylabel('CDF')
    ax0.set_title('CDF of Total Time, Poisson Intensity: '+str(args.intensity))
    ax0.grid(True)

    ax1 = plt.subplot(top_gs[1])
    ax1.plot(sorted_queue_time, np.arange(len(sorted_queue_time))/len(sorted_queue_time), marker='o', markersize=2)
    ax1.set_xlabel('Queue Time (ms)')
    ax1.set_ylabel('CDF')
    ax1.set_title('CDF of Queue Time')
    ax1.grid(True)

    ax2 = plt.subplot(top_gs[2])
    ax2.plot(sorted_infer_time, np.arange(len(sorted_infer_time))/len(sorted_infer_time), marker='o', markersize=2)
    ax2.set_xlabel('Inference Time (ms)')
    ax2.set_ylabel('CDF')
    ax2.set_title('CDF of Inference Time')
    ax2.grid(True)

    ax3 = plt.subplot(bottom_gs[0])
    ax3.plot(sorted_ee_heads, np.arange(len(sorted_ee_heads))/len(sorted_ee_heads), marker='o', markersize=2)
    ax3.set_xlabel('EE Head')
    ax3.set_ylabel('CDF')
    ax3.set_title('CDF of EE Head')
    ax3.grid(True)
    
    ax4 = plt.subplot(bottom_gs[1])
    ax4.plot(np.arange(len(total_time_list)), total_time_list, marker='o', markersize=2)
    ax4.set_xlabel('Task Index (sorted)')
    ax4.set_ylabel('Total Time (ms)')
    ax4.set_title('Total Time Series, Poisson Intensity: '+str(args.intensity))
    ax4.grid(True)
    
    
     
    plt.tight_layout()
    plt.savefig(f'./single_model_result/total_time_cdf_intensity_{args.intensity}.png')
    
