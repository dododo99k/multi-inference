
# Scheduler with history info
import time,copy,os,pickle,argparse,random,math,shutil,queue,tqdm
from collections import deque
from torch.multiprocessing import Pool,Process, Semaphore, Event, Manager, Value, Queue, current_process
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import torch
from functions import Task, FastRollingWindow, InferenceRecord
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
        self.gap_time = 3 # ms, 1ms for inference time variance, 2ms for H2D and D2H time usage
        self.poisson_intensity = 0.0 # estimated poisson intensity for task arrival
        self.poisson_distribution = []
        
        self.time = 0
        self.batch_id = 0
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

    def schedule(self, debug_force_put_into_gpu=False): # single model scheduling
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
            print(f'[Set DDL] time \033[34m{self.time}\033[0m ms, task id \033[33m{_first_task.tk_id}\033[0m, Updated earliest ddl: {self.buffer_queue_earliest_ddl} ms, first arrive time: {self.buffer_queue_1st_arrive_time} ms')
            
            
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
            task.batch_id = self.batch_id
            # put data into shared memory
            # self.shared_mem_list[self.shared_mem_index.value].copy_(task.input_data.squeeze(0))
            # self.shared_mem_index.value += 1
            self.execute_queue.put(task)
        
        self.permit.set()  # notify the worker to start processing
        try:
            print(f'[Execute] time \033[34m{self.time}\033[0m ms, \033[34m{self.batch_id}\033[0m batch, scheduled \033[34m{buffer_queue_size}\033[0m tasks to GPU, ee head: \033[32m{self.ee_head.value}\033[0m, with run time {self.model_profile[buffer_queue_size][self.ee_head.value]:.4f} ms')
        except Exception as e:
            print(f'[Execute] time \033[34m{self.time}\033[0m ms, \033[34m{self.batch_id}\033[0m batch, scheduled Extra Large batch of \033[31m{buffer_queue_size}\033[0m tasks to GPU, ee head: \033[32m{self.ee_head.value}\033[0m')
        self.buffer_queue_earliest_ddl = None
        self.buffer_queue_1st_arrive_time = None
        self.last_time_slot_buffer_size = 0
        self.batch_id += 1
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
                task.finalize(if_print=False)
                tasks_to_finish.append(task)
            except queue.Empty:
                break
        if len(tasks_to_finish) == 0:
            return
        
        _first_task = tasks_to_finish[0]

        print(f'[Finish ] time \033[34m{self.time}\033[0m ms, \033[34m{_first_task.batch_id}\033[0m batch, first task id \033[33m{_first_task.tk_id}\033[0m, ee_head: \033[32m{_first_task.ee_head}\033[0m, inference time {_first_task.infer_time:.4f}')
        # dynamic check missed slo and adjust gap_time
        # miss_slo = False
        # for task in tasks_to_finish:
        #     if (task.finish_time - task.arrive_time) > task.slo:
        #         missed_slo = True
        #         break
        # if miss_slo:
        #     self.gap_time = min(self.max_gap, self.gap_time + 5.0)
        # else:
        #     self.gap_time = max(self.min_gap, self.gap_time - 0.1)
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
    
