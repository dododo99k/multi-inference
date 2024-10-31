import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob


model_names = ['resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg19']

length = 1000

model = model_names[0]

all_latencies = []

for i in range(1,6):

    adding = 'parallel_'+str(i)+'_'

    save_name = str(model)+'_'+adding+str(length)


    FilenamesList = glob.glob('./profile_results/figures/result_'+save_name+'*.pkl')

    latencies = []
    for file in FilenamesList:
        latencies += pickle.load(open(file,'rb'))
    latencies = 1000*np.array(latencies)

    all_latencies.append(latencies)

# all_latencies = np.array(all_latencies)


# fig = plt.figure(figsize =(10, 7))
# ax = fig.add_axes([0, 0, 1, 1])
plt.boxplot(all_latencies)
plt.xlabel('number of parallel runs')
plt.ylabel('inference latency (ms)')
plt.title(model)
plt.savefig('./profile_results/figures/parallel_run_latency'+model+'.png')
plt.show()

print('done')
