import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
import random

random.seed(0)
np.random.seed(0)

def add_noise_within_circle(data, center, distance):
    noisy_data = data.copy()
    noisy_data += distance*0.001
    num_samples, num_features = data.shape

    noise = np.random.uniform(-1, 1, size=(num_samples, num_features))
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    noise = (noise / noise_norm)

    random_scale = pow(10, np.log10(noisy_data))
    scaled_noise = noise * random_scale

    noisy_data += scaled_noise
    noisy_data = np.clip(noisy_data, 0, 1)
    
    return pd.DataFrame(noisy_data, columns=data.columns)

def add_noise_outside_circle(data, center, distance):
    noisy_data = data.copy()
    num_samples, num_features = data.shape

    noise = np.random.uniform(-1, 1, size=(num_samples, num_features))
    distance_new = np.reshape(distance, (1, num_features))
    noise_norm = np.linalg.norm(noise, axis=1, keepdims=True)
    
    distance_new = np.reshape(distance, (1, num_features))
    noise = (noise / noise_norm) * distance_new

    noisy_data += noise
    return noisy_data

names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country']
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital loss', 'hours-per-week']
cat_features = [col for col in names if col not in numerical_features]

data_path = "/nfsdata/tianhao/dataset/UCLAdult/noise/nopca/UCLAdult_norm105_replace_0.csv"
original_data = pd.read_csv(data_path, skipinitialspace=True)
print("original_data.shape", original_data.shape)
print("original_data.head\n", original_data.head())

# cols = original_data.columns.tolist()
# print("cols", cols)
numerical_features = ['0', '1', '2', '3', '4', '5']

log_resp_path = "/nfsdata/tianhao/cluster_results/UCLAdult_norm105/nopca/noise/original_data/k-means++/0/data/class_proportions1_log_resp_50.npy"
log_resp = np.load(log_resp_path)
resp = np.exp(log_resp)
print("resp.shape", resp.shape)

cluster_result = np.argmax(resp, axis=1)
sc_idxs = np.where(cluster_result==0)[0]
print("sc_idxs.shape", sc_idxs.shape)
dt_idxs = np.where(np.isin(cluster_result, [1, 2, 3, 4]))[0]
print("dt_idxs.shape", dt_idxs.shape)

sc_data = original_data.loc[sc_idxs, numerical_features]
sc_center = np.mean(sc_data, axis=0)
iner_distance = np.mean(np.square(sc_data - sc_center), axis=0)
print("iner_distance", iner_distance)

# sc_data_noisy_inner = add_noise_within_circle(sc_data, sc_center, iner_distance)
# original_data.loc[sc_idxs, numerical_features] = sc_data_noisy_inner
# iner_distance_new = np.mean(np.square(sc_data_noisy_inner - sc_center), axis=0)
# print("iner_distance_new", iner_distance_new)

sc_data_noisy_inner = add_noise_outside_circle(sc_data, sc_center, iner_distance)
original_data.loc[sc_idxs, numerical_features] = sc_data_noisy_inner
iner_distance_new = np.mean(np.square(sc_data_noisy_inner - sc_center), axis=0)
print("iner_distance_new", iner_distance_new)

inner_data_path = data_path.replace("UCLAdult_norm105_replace_0.csv", "UCLAdult_norm105_iner.csv")
original_data.to_csv(inner_data_path, index=False)

dt_data = original_data.loc[dt_idxs, numerical_features]
oter_distance = np.mean(np.square(dt_data - sc_center), axis=0)
print("oter_distance", oter_distance)

sc_data_noisy_outer = add_noise_outside_circle(sc_data, sc_center, oter_distance)
original_data.loc[sc_idxs, numerical_features] = sc_data_noisy_outer
oter_distance_new = np.mean(np.square(sc_data_noisy_outer - sc_center), axis=0)
print("oter_distance_new", oter_distance_new)

outer_data_path = data_path.replace("UCLAdult_norm105_replace_0.csv", "UCLAdult_norm105_oter.csv")
original_data.to_csv(outer_data_path, index=False)

sc_data_noisy_outer = add_noise_outside_circle(sc_data, sc_center, oter_distance*10)
original_data.loc[sc_idxs, numerical_features] = sc_data_noisy_outer
oter_distance_new = np.mean(np.square(sc_data_noisy_outer - sc_center), axis=0)
print("oter_distance_new", oter_distance_new)

outer_data_path = data_path.replace("UCLAdult_norm105_replace_0.csv", "UCLAdult_norm105_oter10.csv")
original_data.to_csv(outer_data_path, index=False)

(dev) [tianhao@headnode transformation]$ python adult_shift.py 
original_data.shape (48842, 105)
original_data.head
           0         1  ...  native-country_Vietnam  native-country_Yugoslavia
0  0.301370  0.044131  ...                       0                          0
1  0.452055  0.048052  ...                       0                          0
2  0.287671  0.137581  ...                       0                          0
3  0.493151  0.150486  ...                       0                          0
4  0.150685  0.220635  ...                       0                          0

[5 rows x 105 columns]
resp.shape (48842, 50)
sc_idxs.shape (113,)
dt_idxs.shape (3886,)
iner_distance 0    0.027662
1    0.004865
2    0.026009
3    0.000314
4    0.002547
5    0.008050
dtype: float64
iner_distance_new 0    0.028184
1    0.004873
2    0.025966
3    0.000314
4    0.002561
5    0.008004
dtype: float64
oter_distance 0    0.052469
1    0.004683
2    0.063026
3    0.025808
4    0.017944
5    0.023545
dtype: float64
oter_distance_new 0    0.027197
1    0.004871
2    0.024777
3    0.000478
4    0.002614
5    0.008099
dtype: float64
oter_distance_new 0    0.077159
1    0.005320
2    0.095597
3    0.010016
4    0.008974
5    0.017560
dtype: float64

