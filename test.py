import matplotlib.pyplot as plt
import os, pickle

record = pickle.load(open('././resnet50 inference time.pkl','rb'))

plt.figure(figsize=(10, 5))
plt.plot(record, marker='o', linestyle='-', color='r')
plt.title('inference latency')
plt.xlabel("inference sequence")
plt.ylabel("time")
plt.savefig('resnet50_inference.jpg')