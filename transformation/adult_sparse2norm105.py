import time
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# 1. Read data
print("Start reading data")
input_path = '../dataset/UCLAdult/UCLAdult.data'
names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country', 'result']
original_data = pd.read_csv(input_path, names = names, skipinitialspace=True)
print("original_data.shape", original_data.shape)
print("original_data.head", original_data.head())

sparse_path = '../dataset/UCLAdult/UCLAdult_sparse.data'
sparse_data = pd.read_csv(sparse_path, skipinitialspace=True)
print("sparse_data.shape", sparse_data.shape)
print("sparse_data.head", sparse_data.head())

# 2. Preprocess data
n = sparse_data.shape[0]
d = 123-5-5-5-2-2-5+6
data = np.zeros((n, d))
minmax_scaler = MinMaxScaler()
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital loss', 'hours-per-week']
numerical_data = minmax_scaler.fit_transform(original_data[numerical_features])
print("numerical_data.shape", numerical_data.shape)
print("numerical_data.head", numerical_data[0:5])
sparse_data = np.array(sparse_data)
data[:, 0] = np.array(numerical_data[:, 0])     # age
data[:, 1:9] = np.array(sparse_data[:, 5:13])   # workclass6-13
data[:, 9] = np.array(numerical_data[:, 1])     # fnlwgt
data[:, 10:26] = np.array(sparse_data[:, 18:34])  # education19-34
data[:, 26] = np.array(numerical_data[:, 2])    # education-num
data[:, 27:34] = np.array(sparse_data[:, 39:46])  # marital-status40-46
data[:, 34:48] = np.array(sparse_data[:, 46:60])  # occupation47-60
data[:, 48:54] = np.array(sparse_data[:, 60:66])  # relationship61-66
data[:, 54:59] = np.array(sparse_data[:, 66:71])  # race67-71
data[:, 59:61] = np.array(sparse_data[:, 71:73])  # sex72-73
data[:, 61] = np.array(numerical_data[:, 3])    # capital-gain
data[:, 62] = np.array(numerical_data[:, 4])    # capital loss
data[:, 63] = np.array(numerical_data[:, 5])    # hours-per-week
data[:, 64:105] = np.array(sparse_data[:, 82:123])  # native-country83-123
print("data.shape", data.shape)
print("data.head", data[0:5])


# 3. Save data
output_path = '../dataset/UCLAdult/UCLAdult_norm105.data'
header = [str(i) for i in range(d)]
with open(output_path, 'w') as f:
    f.write(",".join(header)+"\n")
data = pd.DataFrame(data)
data.to_csv(output_path, mode='a', index=False, header=False)
print("Save data to", output_path)
