import matplotlib.pyplot as plt
import numpy as np
import pickle
from functions import Task
from collections import OrderedDict


intensity = 0.4
num_proc  = 5
model_names = ['resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg19']

# ids = []
# for t in range(10000):
#     np.random.seed(t)
#     ids.append(np.random.randint(num_proc))
# plt.plot(ids)
# plt.show()

all_results = pickle.load(open(f'./heteo_result/results_batch_{intensity}_{num_proc}.pkl','rb'))

# results = OrderedDict(sorted(unsorted_results.items()))

# ids = [r.model_id for r in results.values()]
# plt.plot(ids)
# plt.show()
latencies_all = []
latencies_combined = {}

for key, val in all_results.items():

    queue_latencies = 1000*np.array([float(r.queue_time) for r in val.values()])
    infer_latencies = 1000*np.array([float(r.infer_time) for r in val.values()])

    queue_latencies = queue_latencies[100:] # remove warm up parts
    infer_latencies = infer_latencies[100:] # remove warm up parts

    latencies_combined[key] = queue_latencies + infer_latencies
    latencies_all += list(latencies_combined[key])

    # plt.switch_backend('agg')
    plt.plot(queue_latencies, label='Queue latencies')
    plt.plot(infer_latencies, label='Inference latencies') #;plt.show()
    plt.legend()
    plt.title(key)
    plt.ylim(0,20)
    plt.xlabel('Index')
    plt.ylabel('Task')
    # plt.show()
    plt.savefig(f'./heteo_result/batch_{key}_{intensity}_{num_proc}_time.jpg')
    plt.close()


    plt.ecdf(queue_latencies, label='Queue latencies')
    plt.ecdf(infer_latencies, label='Inference latencies') #;plt.show()
    plt.legend()
    plt.title(key)
    plt.xlim(0,20)
    plt.xlabel('Latency (ms)')
    plt.ylabel('CDF')
    # plt.show()
    plt.savefig(f'./heteo_result/batch_{key}_{intensity}_{num_proc}_cdf.jpg')
    plt.close()


# plot ecdf for all models
for name in model_names:
    plt.ecdf(latencies_combined[name], label=name)
# plt.ecdf(latencies_all, label='All in All')
plt.legend()
plt.xlim(0,100)
plt.title('intensity: '+str(intensity))
plt.xlabel('Queue and Compute Latency (ms)')
plt.ylabel('CDF')
# plt.show()
plt.savefig(f'./heteo_result/batch_all_{intensity}_{num_proc}_cdf.jpg')
plt.close()


print('done')













