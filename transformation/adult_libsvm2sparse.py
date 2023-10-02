# GaussianMixture Clustering
import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# 1. Read data
print("Start reading data")

input_path = '../dataset/UCLAdult/UCLAdult_libsvm.data'
with open(input_path, "r") as file:
    lines = file.readlines()
n = len(lines)
d = 123
data = np.zeros((n, d))
labels = np.zeros((n, 1))
print("data.shape", data.shape)
for i in range(n):
    line = lines[i]
    line = line.strip()
    label = int(line.split(' ')[0])
    labels[i] = label if label == 1 else 0
    line = line.split(' ')[1:]
    for element in line:
        dimension = int(element.split(':')[0])-1
        if element.split(':')[1] != '1':
            print("*********************error*********************")
            print("i", i, "dimension", dimension)
        data[i][dimension] = 1
print("data.shape", data.shape)

sparse_path = '../dataset/UCLAdult/UCLAdult_sparse.data'
header = [str(i) for i in range(d)]
with open(sparse_path, 'w') as f:
    f.write(",".join(header)+"\n")
data = pd.DataFrame(data)
data.to_csv(sparse_path, mode='a', index=False, header=False)
np.savetxt('../dataset/UCLAdult/UCLAdult.labels', labels, delimiter=",", fmt="%d")
print("save to", sparse_path)