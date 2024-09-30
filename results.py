# import numpy as np
# import matplotlib.pyplot as plt

# result_1 = 112.5633685850189

# result_2 = [110.71183619799558, 109.67995051300386]

# result_3 = [126.42417698696954, 126.51685032102978, 126.411244133953]

# result_4 = [171.36167771095643, 171.78704009501962, 170.98292579699773, 170.28801876300713]

# result_8 = [333.04086198302684, 334.77299384900834, 335.25418289599475, 336.01504508702783, 334.8496183420066, 335.2468893970363, 335.2468893970363, 328.7138135530404]

# result_12 = [483.05019710800843, 486.62833106098697, 496.4638547550421, 500.3521261130227, 502.20310785004403, 502.3081440909882, 502.9229678559932, 504.61974529502913, 497.4537989710225, 499.076673965028, 500.3895437520114, 501.331232578028]


# data = [result_1, result_2, result_3, result_4, result_8, result_12]

# labels = ['1', '2', '3', '4', '8', '12']

# plt.boxplot(data, labels = labels)
# plt.yscale('log')
# plt.xlabel('concurrency')
# plt.ylabel('latency')
# plt.savefig('latency_concurrency_box.png')
# plt.show()


import os, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = ['1', '2', '4', '8', '12']

data = []

for process in labels:
    # read results from one dir
    print(f'process number: {process}')
    files = sorted(os.listdir(str(process)), key=len)
    result = []
    for file_name in files:
        # print('./'+str(process)+'/'+file_name)
        record = pickle.load(open('./'+str(process)+'/'+file_name,'rb'))
        result += record[1:]
    
    df = pd.DataFrame(result, columns=['Value'])
    Q1 = df['Value'].quantile(0.25)
    Q3 = df['Value'].quantile(0.75)
    IQR = Q3 - Q1

    # upper bound and lower bound
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # outliers
    outliers = df[(df['Value'] < lower_bound) | (df['Value'] > upper_bound)]
    print(f'outliers number: {len(outliers)}')

    data.append(result)


# box

# plt.boxplot(data, labels = labels, whis=100)
# plt.yscale('log')
# plt.xlabel('concurrency with CUDA MPS')
# plt.ylabel('latency')
# plt.savefig('latency_concurrency_box with CUDA MPS.png')

# mean var

means = [np.mean(sublist) for sublist in data]
variances = [np.var(sublist) for sublist in data]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(means, marker='o', linestyle='-', color='b')
plt.title('Means of latency')
plt.xlabel('Concurrency')
plt.ylabel('Sec')

plt.xticks(ticks=range(len(means)), labels=labels)

# Plotting the variances
plt.subplot(1, 2, 2)
plt.plot(variances, marker='o', linestyle='-', color='r')
plt.title('Variances of latency')
plt.xlabel('Concurrency')
plt.ylabel('Variance Value')
plt.xticks(ticks=range(len(means)), labels=labels)
# Show plots
plt.savefig('latency_concurrency mean_var.png')
