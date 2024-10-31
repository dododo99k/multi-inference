import time,copy,os,pickle,argparse,random,math,shutil,queue,tqdm
from torch.multiprocessing import Pool, Manager, current_process
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from torchvision.models import resnet50, ResNet50_Weights


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


def load_official_model_dataset():

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()

    torch.manual_seed(1)
    dataset = torch.randn(1, 3, 224, 224).to(device) 

    return model, dataset


def duplicate_model_dataset(model, dataset, num_proc):

    models, datasets = [model.to(device)], [dataset.to(device)]
    for i in range(num_proc-1):
        mdl = copy.deepcopy(model)
        mdl.to(device)
        models.append(mdl)

        data = copy.deepcopy(dataset)
        data.to(device) # TODO this code does not make the data to GPU, why?
        datasets.append(data)

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

        task.start_time = time.time()

        outputs = model(data)

        outputs.detach().cpu()# .numpy().argmax() # model.eval() is already set, here to device is still needed, TODO XXX

        task.finish_time = time.time()

        task.finalize(if_print=False)

        finish_queue.put(task)

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
    
    argparser.add_argument('-l','--duration',
                           type = int,
                           default=3000,
                           help='time duration in ms')
    
    argparser.add_argument('-pr','--num_process',
                           type = int,
                           default=1,
                           help='time duration in ms')

    
    args = argparser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.multiprocessing.set_start_method('spawn')
    num_proc = args.num_process
    tid = 0

    # model_name = 'resnet50' # or resnet101
    # models, datasets = load_homo_model_dataset(model_name, num_proc) 
    model, dataset = load_official_model_dataset()
    models, datasets = duplicate_model_dataset(model, dataset, num_proc)

    # Create a manager to manage shared data
    with Manager() as manager:
        print('enter in manager')
        buffer_queue = manager.Queue() # this queue is to buffer all the incoming tasks, before they are sent to execute_queue, the scheduler mainly schedule this queue
        execute_queue = manager.Queue() # this queue is to send tasks to workers, XXX once task is put, workers will immediately get it to execute XXX
        finish_queue = manager.Queue() # this queue is to collect stats

        # Create a persistent pool of worker processes 
        pool = Pool(processes=num_proc)
        
        # Start worker processes to process tasks in the queue
        for i in range(num_proc):
            pool.apply_async(worker, args=(execute_queue, finish_queue, models[i], datasets[i])) # workers see execute_queue, not buffer_queue
        
        start_time = time.time()

        # main loop to receive tasks
        for t in range(args.duration):

            task = Task(tid=tid, arrive_time=time.time())
            # print(task.tid, task.arrive_time)
            buffer_queue.put(task) # always put into the buffer queue

            ## XXX main scheduling here XXX #######
            schedule(buffer_queue, execute_queue) # TODO this is a native scheduler, TBD

            tid += 1 # increas task id

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


    pickle.dump(results, open(f'./homomodel_result/results_{num_proc}.pkl','wb'))

    total_time = time.time() - start_time
    print('time usage:', total_time)


    ##### plot figure #######
    # results = pickle.load(open('results.pkl','rb'))
    queue_times = [float(r.queue_time) for r in results.values()]
    infer_times = [float(r.infer_time) for r in results.values()]
    # for i in range(999,-1,-1):
    #     if queue_times[i]> 0.05:
    #         break

    print(np.mean(infer_times), np.std(infer_times))
    # plt.switch_backend('agg')
    plt.plot(queue_times, label='Queue Times')
    plt.plot(infer_times, label='Inference Times') #;plt.show()
    plt.legend()
    plt.ylim(0,0.02)
    plt.xlabel('Index')
    # plt.yticks(np.arange(0,0.15, 0.01))
    plt.ylabel('Time')
    # fname = f'{args.intensity}_{num_proc}.jpg'
    plt.savefig('./test_queue_proccess'+str(num_proc)+'_'+str(int(np.mean(infer_times)*10000))+'_'+str(int(np.std(infer_times)*10000))+'.jpg')
    plt.close()
    # plt.show()