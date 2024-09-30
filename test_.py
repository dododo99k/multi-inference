from torch.multiprocessing import Process, set_start_method
import torch
import time


stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
torch.cuda.synchronize()
def process1():
    global stream1, stream2
    with torch.cuda.stream(stream1):
        print("IM HERE 1\n")
        print(time.time(),"time in process 1")
        time.sleep(5)
def process2():
    global stream1, stream2
    with torch.cuda.stream(stream2):
        print("IM HERE 2\n")
        print(time.time(),"time in process 2")
        time.sleep(5)
if __name__ == "__main__":
    set_start_method('spawn',force = True)
    start = time.time()
    p1 = Process(target = process1)
    p2 = Process(target = process2)
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    torch.cuda.synchronize()
    print("Time for parallel implementation: {}".format(time.time() - start))