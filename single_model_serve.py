import time,copy,os,pickle,argparse,random,math,shutil,queue,tqdm,json,gc
from collections import deque
from torch.multiprocessing import Pool,Process, Semaphore, Event, Manager, Value, Queue, current_process
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from functions import Task, FastRollingWindow, InferenceRecord
from ee_models import *
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights, vgg11, VGG11_Weights, vgg19, VGG19_Weights

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
    random_data = torch.randn(30, 3, 224, 224).to(dev)
    
    # warm up
    for datasize in range(1, 31):
        # data = shared_mem_list[:datasize].to(dev)
        data = random_data[:datasize]
        for i in range(5):
            for ee in range(0, 14):
                _ = model(data, ee)
    print(f"batch {current_process().name} warmed up.", flush=True)
    
    
    
    # serving loop
    while True:
        permit.wait()           # 阻塞等待许可（不占 CPU）
        if stop.is_set():          # 允许优雅收尾
            break
        receive_time_temp = time.time_ns()
        # print(f"{current_process().name}, shared memory pointer {data_number.value}, ee_head {ee_head.value}")
        torch.cuda.synchronize()  # 确保数据已传输到 GPU
        # data = shared_mem_list[:data_number.value].to(dev)
        if execute_queue.qsize() != data_number.value:
            print(f"[WARNING] {current_process().name} execute_queue size {execute_queue.qsize()} not equal to data_number {data_number.value}")
            
        data = random_data[:data_number.value]
        
        start_time_temp = time.time_ns()
        results = model(data, int(ee_head.value))  # 批处理推理
        
        torch.cuda.synchronize()  # 确保推理完成
        finish_time_temp = time.time_ns()
        # results.detach().cpu()     # 释放许可，通知主进程批处理已完成
        
        tmp = []
        while True:
            try:
                task = execute_queue.get_nowait()
            except queue.Empty:
                break
            task.start_time = (start_time_temp - task.base_time) / 1_000_000
            task.finish_time = (finish_time_temp - task.base_time) / 1_000_000
            tmp.append(task)
        for task in tmp:
            execute_queue.put(task)
        
        permit.clear()            # 重置许可,等待下一批数据
        done.set()

        # Disable print to avoid I/O blocking that causes ~70ms delay
        # print(f"{current_process().name} finished processing batch of size {len(tmp)}, time: {(time.time_ns() - receive_time_temp) / 1_000_000} ms, inference time: {(finish_time_temp - start_time_temp) / 1_000_000} ms.")


# Scheduler with history info
class Scheduler():
    def __init__(self, num_proc, buffer_queue, execute_queue, finish_queue, shared_mem_list, shared_mem_index, permit, done, stop, model_profile, ee_head):
        self.num_proc = num_proc
        self.buffer_queue = buffer_queue
        self.execute_queue = execute_queue
        self.finish_queue = finish_queue
        self.shared_mem_list = shared_mem_list
        self.shared_mem_index = shared_mem_index
        ########################
        self.buffer_queue_1st_arrive_time = None # earliest arrive time in the buffer queue (batch)
        self.buffer_queue_earliest_ddl = None # earliest ddl in the buffer queue (batch)
        self.single_model_id = 0 # only serve single model
        
        ######## some synchronization variables ########
        self.permit = permit
        self.done = done
        self.stop = stop
        # self.finish_times = finish_times
        
        # model profile for scheduling decision
        # 2 diemension array: model_profile[batch size+1][early exit head] = inference time
        # +1 because batch size from 0 to N
        self.model_profile = model_profile
        self.ee_head = ee_head
        self.gap_time = 15 # ms
        self.poisson_intensity = 0.0 # estimated poisson intensity for task arrival
        self.poisson_distribution = []
        
        self.time = 0
        self.buffer_history = []
        self.last_time_slot_buffer_size = 0
        self.executed_estimation = False

    def simple_schedule(self):
        self.time += 1
        for proc in range(self.num_proc): # for each proc
            if len(self.buffer_queues[proc]) == 0:
                pass
            else:
                task = self.buffer_queues[proc].popleft()
                self.execute_queues[proc].append(task)

    def schedule(self): # single model scheduling
        # this scheduling action happens for all time intervals, if not action needs to be done, then basically skip
        start_time = time.perf_counter_ns()
        self.time += 1
        
        # Store how many tasks arrived in this time slot
        if len(self.buffer_queue) > 0 and len(self.buffer_queue) - self.last_time_slot_buffer_size >= 0:
            self.buffer_history.append(len(self.buffer_queue) - self.last_time_slot_buffer_size)
        elif self.last_time_slot_buffer_size > len(self.buffer_queue):
            self.buffer_history.append(len(self.buffer_queue))
        else: # no task arrived
            self.buffer_history.append(0)
        self.last_time_slot_buffer_size = len(self.buffer_queue)
        
        # Set earliest ddl for current batch if not set
        if self.buffer_queue_earliest_ddl is None and self.buffer_queue:
            _first_task = self.buffer_queue.popleft()
            self.buffer_queue_1st_arrive_time = _first_task.arrive_time
            self.buffer_queue_earliest_ddl =  _first_task.arrive_time + _first_task.slo # just get the first task's slo as the earliest ddl
            self.buffer_queue.appendleft(_first_task)
            print(f'[Set DDL] time \033[34m{self.time}\033[0m ms, Updated earliest ddl: {self.buffer_queue_earliest_ddl} ms, first arrive time: {self.buffer_queue_1st_arrive_time} ms')
            
            
        batch_size = len(self.buffer_queue)
        if self.permit.is_set(): # GPU worker is still busy
            return None
        elif len(self.buffer_queue) > 0: # free GPU worker and has tasks in buffer
            if self.executed_estimation == False: # not executed estimation yet, only do once when gpu is free
                self.est_slot_num , self.est_batch_threshold, self.est_ee_head = self.estimate_slot_num_for_batch()
                # print(f'[DEBUG] Poisson {self.poisson_intensity}, Estimated slot num: {self.est_slot_num}, estimated batch size threshold: {self.est_batch_threshold}, estimated ee head: {self.est_ee_head}')

            # print(f'[DEBUG] time {self.time} ms, buffer size: {len(self.buffer_queue)}, earliest ddl: {self.buffer_queue_earliest_ddl}')
            # print(f'[DEBUG] gap_time: {self.gap_time}, estimated batch time: {self.model_profile[batch_size][0]}')
            
            ##########
            if len(self.buffer_queue) >= self.model_profile.shape[0]: # exceed max batch size
                self.ee_head.value = int(self.model_profile.shape[1]-1) # use earliest exit head
                self.put_tasks_into_gpu(if_meet_deadline=False)
                return None
            
            if self.buffer_queue_earliest_ddl < float(self.time + self.gap_time) + self.model_profile[batch_size][0]: # exceed deadline
                # decide to allocate tasks to GPU
                # search ee_head to minimize deadline miss
                for head in range(self.model_profile.shape[1]):
                    if self.model_profile[batch_size][head] < self.buffer_queue_earliest_ddl - self.time:
                        self.ee_head.value = head
                        break
                if self.ee_head.value == -1: # cant meet deadline even with full model
                    self.ee_head.value = int(self.model_profile[batch_size][-1])  # use earliest exit head
                self.put_tasks_into_gpu(if_meet_deadline=True)
                return None
                
            elif False: # not meet deadline
                if len(self.buffer_queue) >= self.est_batch_threshold:
                    self.ee_head.value = self.est_ee_head
                    self.put_tasks_into_gpu(if_meet_deadline=False)
                    return
            


    def put_tasks_into_gpu(self, if_meet_deadline=True):
        buffer_queue_size = len(self.buffer_queue)
        self.shared_mem_index.value = len(self.buffer_queue) # set shared memory index
        while not len(self.buffer_queue) == 0:
            task = self.buffer_queue.popleft()
            task.batch_meet_deadline = if_meet_deadline
            # put data into shared memory
            # self.shared_mem_list[self.shared_mem_index.value].copy_(task.input_data.squeeze(0))
            # self.shared_mem_index.value += 1
            self.execute_queue.put(task)
            
        self.permit.set()  # notify the worker to start processing
        try:
            print(f'[Execute] time \033[34m{self.time}\033[0m ms, scheduled \033[34m{buffer_queue_size}\033[0m tasks to GPU, ee head: \033[32m{self.ee_head.value}\033[0m,with run time {self.model_profile[buffer_queue_size][self.ee_head.value]} meet deadline: {if_meet_deadline}')
        except Exception as e:
            print(f'[Execute] time \033[34m{self.time}\033[0m ms, scheduled Extra Large batch of \033[31m{buffer_queue_size}\033[0m tasks to GPU, ee head: \033[32m{self.ee_head.value}\033[0m, meet deadline: {if_meet_deadline}')
        self.buffer_queue_earliest_ddl = None
        self.buffer_queue_1st_arrive_time = None
        self.last_time_slot_buffer_size = 0
        self.done.clear()
    
    def finish_tasks_from_gpu(self):
        # see which model has finished its batch processing
        # Non-blocking check: only process if GPU is done, otherwise skip immediately
        if not self.done.is_set():
            return  # GPU still busy, skip
        
        self.shared_mem_index.value = 0 # reset shared memory index
        self.ee_head.value = -1 # set to default
        
        # Optimized: collect all tasks first, then append in batch
        tasks_to_finish = []
        while True:
            try:
                task = self.execute_queue.get_nowait()
                tasks_to_finish.append(task)
            except queue.Empty:
                break
        
        # Batch append to finish_queue
        self.finish_queue.extend(tasks_to_finish)
        
        self.executed_estimation = False # reset estimation flag for next scheduling round

    def stop_all(self):
        self.permit.set()
        self.stop.set()
    
    def estimate_poisson_intensity(self):
        window_length = min(1000, len(self.buffer_history))
        self.poisson_intensity = np.mean(self.buffer_history[-window_length:])

    def get_poisson_distribution(self, k_max=None):
        if k_max is None:
            k_max = int(self.poisson_intensity + 5 * self.poisson_intensity**0.5)
        ks = np.arange(0, k_max + 1)
        self.poisson_distribution = np.exp(-self.poisson_intensity) * np.power(self.poisson_intensity, ks) / np.array([math.factorial(k) for k in ks])
        if len(self.poisson_distribution) > self.model_profile.shape[0]:
            self.poisson_distribution = self.poisson_distribution[:self.model_profile.shape[0]]
        
    def get_n_slot_estimate_batch_size(self, slot_num):
        delta_t = 1.0 # ms
        lam_tot = self.poisson_intensity * delta_t * slot_num
        k_max = int(lam_tot + 5 * math.sqrt(lam_tot))
        ks = np.arange(0, k_max + 1)
        _distribution = np.exp(-lam_tot) * lam_tot**ks / np.array([math.factorial(k) for k in ks]) # event number distribution in n slots
        # if len(_distribution) > self.model_profile.shape[0]:
        #     _distribution = _distribution[:self.model_profile.shape[0]]
        return _distribution

        
    def estimate_slot_num_for_batch(self):
        # estimate how many time slots are needed to accumulate target_batch_size tasks
        self.executed_estimation = True
        self.estimate_poisson_intensity()
        self.get_poisson_distribution()
        # print(f'[DEBUG] estimated poisson intensity: {self.poisson_intensity} tasks/ms')
        if self.poisson_intensity == 0:
            return None, None, None
        # look for a slot num that can accumulate target_batch_size tasks
        # meet the inference time of xx batch size in n-slot == the buffer waiting time of slot num
        est_inference_time = 0.0
        for i , prob in enumerate(self.poisson_distribution):
            est_inference_time += prob * self.model_profile[i][0]
        
        # n_slot_batch_est_inference_time = 0.0
        for _ee_head in range(self.model_profile.shape[1]): # ee head
            for slot_num in range(1,30+1): # max time slots
                n_slot_batch_est_inference_time = 0.0
                n_slot_batch_size_distribution = self.get_n_slot_estimate_batch_size(slot_num)
                if len(n_slot_batch_size_distribution) > self.model_profile.shape[0]:
                    left_distribution_sum = np.sum(n_slot_batch_size_distribution[self.model_profile.shape[0]:])
                    if left_distribution_sum > 0.1: # too much mass left
                        break
                    n_slot_batch_size_distribution = n_slot_batch_size_distribution[:self.model_profile.shape[0]]
                # search for estimated inference time
                for i , prob in enumerate(n_slot_batch_size_distribution):
                    n_slot_batch_est_inference_time += prob * self.model_profile[i][_ee_head]
                if n_slot_batch_est_inference_time <= slot_num:
                    return slot_num, math.ceil(slot_num * self.poisson_intensity), _ee_head

        return None, None, None
    



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
            
    time.sleep(5)
    for t in range(1000): # extra 1000 ms to process remaining tasks
        loop_start_time = time.perf_counter()
        scheduler.finish_tasks_from_gpu()
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
    
    for proc in range(num_proc):
        results = {}
        while len(finish_queues[proc]) > 0:
            task = finish_queues[proc].popleft()
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
    