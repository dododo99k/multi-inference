import numpy as np



class Task():
    def __init__(self, model_id=-1, tid=-1, arrive_time = -1, start_time = -1, finish_time = -1,):
        # self.model_name = model_name
        self.model_id = model_id
        self.tid = tid
        self.arrive_time = arrive_time
        self.start_time = start_time
        self.finish_time = finish_time
        self.infer_time = -1
        self.queue_time = -1
        self.batchsize = 1

    def finalize(self, if_print=True):
        self.infer_time = self.finish_time - self.start_time
        self.queue_time = self.start_time - self.arrive_time
        if if_print: print('infer time: ', self.infer_time, 'queue time: ', self.queue_time)
        
