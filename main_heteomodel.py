import time,copy,os,pickle,argparse,random,math,shutil,queue,tqdm
from torch.multiprocessing import Pool, Manager, current_process
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from functions import Task

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
def worker(execute_queue, finish_queue, model, dataset):
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

        outputs = model(data)

        outputs.detach().cpu()
        # print(f"{current_process().name} process finish task: {task.tid}")
        task.finish_time = time.perf_counter()

        task.finalize(if_print=False)

        finish_queue.put(task) # put it into the finish queue for stats and analysis later


def schedule(num_proc, buffer_queues, execute_queues):
    # this scheduling action happens for all time intervals, if not action needs to be done, then basically skip
    
    for proc in range(num_proc): # for each proc

        if buffer_queues[proc].empty():
            pass
        else:
            task = buffer_queues[proc].get()
            execute_queues[proc].put(task)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-s','--seed',
                           type=int,
                           default=42,
                           help='random seed')
    
    argparser.add_argument('-pd','--intensity',
                           type = float,
                           default=0.1,
                           help='poisson density for user inference request number per ms')
      
    argparser.add_argument('-l','--duration',
                           type = int,
                           default=30000,
                           help='time duration in ms')
    
    
    args = argparser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method('spawn')

    model_names = ['resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg19']

    # load models and weights and datasets from PyTorch offical achieve
    models, datasets = load_offical_models_datasets()
    num_proc = len(models)
    tid = 0
    sim_interval = 1 #ms
    # total_tasks = []

    # generate traffic 
    traffic = generate_traffic(args.intensity, args.duration) # here, intensity and duration are both in ms units
    print('generated totally', sum(traffic), 'tasks')

    # Create a manager to manage shared data
    with Manager() as manager:
        print('entering the manager')

        # Shared queue (per model) managed by the manager
        buffer_queues, execute_queues, finish_queues = [], [], []
        for q in range(num_proc):
            buffer_queue = manager.Queue() # this queue is to buffer all the incoming tasks, before they are sent to execute_queue, the scheduler mainly schedule this queue
            execute_queue = manager.Queue() # this queue is to send tasks to workers, XXX once task is put, workers will immediately get it to execute XXX
            finish_queue = manager.Queue() # this queue is to collect stats
            buffer_queues.append(buffer_queue)
            execute_queues.append(execute_queue)
            finish_queues.append(finish_queue)

        # Create a persistent pool of worker processes 
        pool = Pool(processes=num_proc)

        # pool.starmap(run_task, [(model, dataset) for dataset in datasets for model in models])
        
        # Start worker processes to process tasks in the queue
        for i in range(num_proc):
            pool.apply_async(worker, args=(execute_queues[i], finish_queues[i], models[i], datasets[i])) # workers see execute_queue, not buffer_queue

        # wait for workers to start
        print('waiting for workers to start...')
        time.sleep(5)

        start_time = time.perf_counter()
        print('starting the main loop...')
        # main loop to receive tasks
        for t in range(args.duration):
            loop_start_time = time.perf_counter()

            for i in range(traffic[t]):
                
                # random request for a model
                idx = np.random.randint(num_proc)

                task = Task(model_id=idx, tid=tid, arrive_time=time.perf_counter())
                # print(task.tid, task.arrive_time)
                buffer_queues[idx].put(task) # always put into the buffer queue

                ## XXX main scheduling here XXX #######
                schedule(num_proc, buffer_queues, execute_queues) # TODO this is a native scheduler, TBD

                tid += 1 # increas task id

            # this control the traffic arrival at real-world time
            elapsed = time.perf_counter() - loop_start_time
            # if elapsed >= 0.001: print('warning, time slot beyond defined unit, time is ', elapsed)
            # while elapsed < 0.001:
            #     elapsed = time.perf_counter() - loop_start_time
            if elapsed < 0.001: # interval is 1ms
                time.sleep(0.001 - elapsed)
            else:
                print('warning, time slot beyond defined unit, time is ', elapsed)

        
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
                results[str(task.tid)] = copy.deepcopy(task)
            all_results[model_names[proc]] = results
                
        # print('collected totally', len(results.keys()), 'tasks')

        # When you're ready to stop the workers, send None into the queue for each worker
        for proc in range(num_proc): 
            execute_queues[proc].put(None)

        # Close and join the pool to wait for workers to finish
        pool.close()
        pool.join()

        print("All tasks processed and workers stopped.")


    pickle.dump(all_results, open(f'./heteo_result/results_{args.intensity}_{num_proc}.pkl','wb'))
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