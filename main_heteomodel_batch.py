import time,copy,os,pickle,argparse,random,math,shutil,queue,tqdm
from collections import deque
from torch.multiprocessing import Pool,Process, Semaphore, Event, Manager, Value, Queue, current_process
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from functions import Task, FastRollingWindow, InferenceRecord

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

from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights, resnet152, ResNet152_Weights, vgg11, VGG11_Weights, vgg19, VGG19_Weights


def load_offical_models_datasets():
    models = []
    # keep the order of append here, which matches that of later part
    models.append(resnet50(weights=ResNet50_Weights.DEFAULT))
    models.append(resnet101(weights=ResNet101_Weights.DEFAULT))
    models.append(resnet152(weights=ResNet152_Weights.DEFAULT))
    models.append(vgg11(weights=VGG11_Weights.DEFAULT))
    models.append(vgg19(weights=VGG19_Weights.DEFAULT))

    datasets = []
    datasets.append(torch.randn(1, 3, 224, 224)) 
    datasets.append(torch.randn(1, 3, 224, 224)) 
    datasets.append(torch.randn(1, 3, 224, 224)) 
    datasets.append(torch.randn(1, 3, 224, 224)) 
    datasets.append(torch.randn(1, 3, 224, 224)) 

    for model in models:
        model.to(device)
        model.eval()

    for dataset in datasets:
        dataset.to(device)

    return models, datasets


# Persistent worker function: Retrieves a task from the queue and processes it
def batch_worker(model, execute_queue, permit, done, stop, shared_mem_list, data_number):
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
    model = model.to(dev).eval()

    print(f"entered batch {current_process().name}. waiting for tasks...", flush=True)
    # warm up
    data = shared_mem_list[:10].to(dev)
    for _ in range(100):
        _ = model(data)

    while True:
        permit.wait()           # 阻塞等待许可（不占 CPU）
        if stop.is_set():          # 允许优雅收尾
            break
        data = shared_mem_list[:data_number.value].to(dev)
        data_number.value = 0      # 重置数据计数器
        results = model(data)
        results.detach().cpu()     # 释放许可，通知主进程批处理已完成
        finish_time_temp = time.time_ns() / 1_000_000
        
        tmp = []
        while True:
            try:
                task = execute_queue.get_nowait()
            except queue.Empty:
                break
            task.finish_time = finish_time_temp
            tmp.append(task)
        for task in tmp:
            execute_queue.put(task)
        
        permit.clear()            # 重置许可，等待下一批数据
        done.set()


# Scheduler with history info
class Scheduler():
    def __init__(self, num_proc, buffer_queues, execute_queues, finish_queues, shared_mem_lists, shared_mem_indexs, permits, dones, stop):
        self.num_proc = num_proc
        self.buffer_queues = buffer_queues
        self.execute_queues = execute_queues
        self.finish_queues = finish_queues
        self.shared_mem_lists = shared_mem_lists
        self.shared_mem_indexs = shared_mem_indexs
        self.permits = permits
        self.dones = dones
        self.stop = stop
        # self.finish_times = finish_times
        
        self.time = 0
        self.running_inference = {}

    def simple_schedule(self, num_proc, buffer_queues, execute_queues):
        for proc in range(num_proc): # for each proc
            if len(buffer_queues[proc]) == 0:
                pass
            else:
                task = buffer_queues[proc].popleft()
                execute_queues[proc].append(task)
    
    def schedule(self):
        # this scheduling action happens for all time intervals, if not action needs to be done, then basically skip
        self.time += 1
        for model_idx, permit in enumerate(self.permits):
            if permit.is_set(): # some worker is still busy
                continue
            
        pass
        # if decide to allocate tasks to GPU, then call put_tasks_into_gpu()
        
        self.put_tasks_into_gpu(model_idx)

    def put_tasks_into_gpu(self, model_idx):
        buffer_queue = self.buffer_queues[model_idx]
        execute_queue = self.execute_queues[model_idx]
        shared_mem = self.shared_mem_lists[model_idx]
        shared_mem_index = self.shared_mem_indexs[model_idx]

        while not len(buffer_queue) == 0:
            task = buffer_queue.popleft()
            # put data into shared memory
            shared_mem[shared_mem_index.value].copy_(task.input_data.squeeze(0))
            shared_mem_index.value += 1
            execute_queue.append(task)
        self.permits[model_idx].set()  # notify the worker to start processing
    
    def finish_tasks_from_gpu(self):
        # see which model has finished its batch processing
        for model_idx, done in enumerate(self.dones):
            if done.is_set():
                # read finished tasks from execute queue and put into finish queue
                execute_queue = self.execute_queues[model_idx]
                finish_queue = self.finish_queues[model_idx]
                while not len(execute_queue) == 0:
                    task = execute_queue.popleft()
                    finish_queue.append(task)
                # clear done event
                self.permits[model_idx].clear() # extra clear for safety
                done.clear()
    
    def stop_all(self):
        for permit in self.permits: 
            permit.set()
        self.stop.set()
        

#### TODO XXX
#### increase the arrival interval, does not have to be 1ms granularity

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-s','--seed',
                           type=int,
                           default=42,
                           help='random seed')
    
    argparser.add_argument('-pd','--intensity',
                           type = float,
                           default=0.4,
                           help='poisson density for user inference request number per ms')
      
    argparser.add_argument('-l','--duration',
                           type = int,
                           default=30000,
                           help='time duration in ms')
         
    argparser.add_argument('-b','--batch',
                           type = bool,
                           default=True,
                           help='if using batch workers')
        
    args = argparser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method('spawn')

    model_names = ['resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg19']

    # load models and weights and datasets from PyTorch offical achieve
    models, datasets = load_offical_models_datasets()
    num_proc = len(models)
    tkid = 0
    sim_interval = 1/1000 #ms
    # total_tasks = []

    # generate traffic 
    traffic = generate_traffic(args.intensity, args.duration) # here, intensity and duration are both in ms units
    print('generated totally', sum(traffic), 'tasks')

    # Create a manager to manage shared data


    # Shared queue (per model) managed by the manager
    buffer_queues = [deque() for _ in range(num_proc)] # this queue is to buffer all the incoming tasks, before they are sent to execute_queue, the scheduler mainly schedule this queue
    execute_queues = [Queue() for _ in range(num_proc)] # this multiprocessing queue is to send tasks to workers, XXX once task is put, workers will immediately get it to execute XXX
    finish_queues = [deque() for _ in range(num_proc)] # this queue is to collect stats

    shared_mem_lists = [torch.empty(10, 3, 224, 224).share_memory_() for _ in range(num_proc)]
    shared_mem_indexs = [Value('i', 0) for _ in range(num_proc)] # indicate the current index to write new data
    permits = [Semaphore(1) for _ in range(num_proc)]
    permits = [Event() for _ in range(num_proc)]  
    dones = [Event() for _ in range(num_proc)]
    stop = Event()

    scheduler = Scheduler(num_proc, buffer_queues, execute_queues, finish_queues, shared_mem_lists, shared_mem_indexs, permits, dones, stop)
    # finish_times = [Value('d', 0.0) for _ in range(num_proc)] # store finish time of each batch worker
    
    
    # Start worker processes to process tasks in the queue
    torch_processes = []
    for i in range(num_proc):
        # if args.batch:
        #     pool.apply_async(batch_worker, args=(execute_queues[i], finish_queues[i], models[i], datasets[i])) # workers see execute_queue, not buffer_queue
        # else:
        #     pool.apply_async(worker, args=(execute_queues[i], finish_queues[i], models[i], datasets[i])) # workers see execute_queue, not buffer_queue
        p = Process(
                target=batch_worker,
                args=(models[i], execute_queues[i], permits[i], dones[i], stop, shared_mem_lists[i], shared_mem_indexs[i]),
                name=f'Worker-{i}'
            )
        p.start()
        torch_processes.append(p)
        
    # wait for workers to start
    print('waiting for workers to start and warm up...')
    time.sleep(5)

    # start_time = time.perf_counter()
    print('starting the main loop...')
    
    #########################
    # generate tasks ahead of simulation
    #########################
    tasks_list = [] # store tasks at each time slot, length should be args.duration
    task_num_per_model_lists = [] # store how many tasks for each model at each time slot
    for t in range(args.duration):
        temp = []
        task_num_per_model_temp = [0]*num_proc
        for i in range(traffic[t]):
            idx = np.random.randint(num_proc)
            task_num_per_model_temp[idx] += 1
            task = Task(model_id=idx, tk_id=tkid)
            tkid += 1 # increas task id
            temp.append(task)
        tasks_list.append(temp)
        task_num_per_model_lists.append(task_num_per_model_temp)
    #########################
    # main loop to receive tasks
    #########################
    base_time = time.time_ns() 
    for t in range(args.duration):
        loop_start_time = time.perf_counter()
        
        task_num_per_model = task_num_per_model_lists[t] # how many tasks for each model at current time slot
        # put tasks into buffer queues
        for task in tasks_list[t]:
            task.arrive_time = (time.time_ns()-base_time) / 1_000_000 
            model_id = task.model_id
            input_data = task.input_data
            # put data into shared memory
            shared_mem_lists[model_id][shared_mem_indexs[model_id]].copy_(input_data.squeeze(0))
            shared_mem_indexs[model_id] += 1
            buffer_queues[model_id].put(task)

        # scheduling
        scheduler.schedule()
        # finish tasks from GPU
        scheduler.finish_tasks_from_gpu()

        # this control the traffic arrival at real-world time
        elapsed = time.perf_counter() - loop_start_time
        if elapsed < sim_interval: # interval is 1 ms = 0.001 s
            time.sleep(sim_interval - elapsed)
        else:
            print('.', end='', flush=True)

    
    # wait for tasks to complete
    print('wait for 5 seconds...')
    time.sleep(5)
    # wait for empty execute queue
    while True:
        empty_all = True
        for execute_queue in execute_queues:
            empty_all = empty_all and execute_queue.empty()
        if empty_all: break
        print('wait for 1 more second...')
        time.sleep(1)

    ###### post processing (XXX this needs to be done here, if the manager() done, then queue are cleared) ######
    all_results = {}
    
    for proc in range(num_proc):
        results = {}
        while not finish_queues[proc].empty():
            task = finish_queues[proc].get()
            # task = task_list[0]
            results[str(task.tk_id)] = copy.deepcopy(task)
        all_results[model_names[proc]] = results
            
    # print('collected totally', len(results.keys()), 'tasks')

    # When you're ready to stop the workers, send None into the queue for each worker
    for proc in range(num_proc): 
        execute_queues[proc].put(None)

    # Close and join the pool to wait for workers to finish

    print("All tasks processed and workers stopped.")


    pickle.dump(all_results, open(f'./heteo_result/results_batch_{args.intensity}_{num_proc}.pkl','wb'))
    print('results are saved')

    # total_time = time.perf_counter() - start_time
    # print('time usage:', total_time)


    # ##### plot figure #######
    # # results = pickle.load(open('results.pkl','rb'))
    # queue_times = [float(r.queue_time) for r in results.values()]
    # infer_times = [float(r.infer_time) for r in results.values()]
    # # for i in range(999,-1,-1):
    # #     if queue_times[i]> 0.05:
    # #         break

    # pass
    # # plt.switch_backend('agg')
    # plt.plot(queue_times, label='Queue Times')
    # plt.plot(infer_times, label='Inference Times') #;plt.show()
    # plt.legend()
    # # plt.ylim(0,0.02)
    # plt.xlabel('Index')
    # # plt.yticks(np.arange(0,0.15, 0.01))
    # plt.ylabel('Time')
    # # fname = f'{args.intensity}_{num_proc}.jpg'
    # plt.savefig(f'./heteo_result/{args.intensity}_{num_proc}.jpg')
    # plt.close()
    # # plt.show()