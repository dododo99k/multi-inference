import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

import pandas



model_names = ['resnet50', 'resnet101', 'resnet152', 'vgg11', 'vgg19']

model = model_names[3]

csvFile = pandas.read_csv('./profile_results/ncu/'+model+'.csv')

names = csvFile.columns

duration = csvFile[names[4]].to_numpy()

throughput = csvFile[names[6]].to_numpy()


cumsums = np.cumsum(duration)

plt.step(cumsums,throughput)
plt.xlabel('kernel duration (in profiling mode)')
plt.ylabel('compute throughput (in profiling mode)')
plt.title(model)
plt.savefig('./profile_results/ncu/model_structure'+model+'.png')
plt.show()

print('done')
