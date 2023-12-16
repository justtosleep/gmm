import numpy as np
import pandas as pd
import argparse
import os

# read command line arguments
parser = argparse.ArgumentParser(description="Sample data from dataset")

# add positional arguments

# add optional arguments
parser.add_argument("--dataset", type=str, help="name of dataset")
parser.add_argument("--data_path", type=str, help="path of dataset")
parser.add_argument("--method", type=str, default="", help="name of dataset processing method")

args = parser.parse_args()
domain = args.dataset
data_path = args.data_path
method = args.method
print("domain: ", domain)
print("data_path: ", data_path)
print("method: ", method)
name = domain+"_"+method if method != "" else domain
data1_path = "/nfsdata/tianhao/dataset/" + data_path.split('.')[0]+"_sample1.data"
# data1_path = "../dataset/" + data_path.split('.')[0]+"_sample1.data"
data = pd.read_csv(data1_path, skipinitialspace=True)
print("data.shape: ", data.shape)
print("data head: ", data.head())


np.random.seed(42)
def add_laplace_noise(data, loc=0.0, scale=1.0):
    noise = np.random.laplace(loc, scale, data.shape)
    noisy_data = data + noise
    return noisy_data

def laplace_noise(data):
    print("add laplace noise to all columns")
    for col in data.columns:
        data[col] = add_laplace_noise(data[col], 0, 1)
    return data

def cluster_shift(data, sc=0, init_param='k-means++', random_seed=987, n_component=50):
    print(f"shift cluster {sc} mean")
    resp_dir = f"/nfsdata/tianhao/cluster_results/{name}/nopca/{init_param}/{random_seed}/data/"
    # resp_dir = f"../cluster_results/{name}/nopca/{init_param}/{random_seed}/data/"
    log_resp1 = np.load(resp_dir+f"class_proportions1_log_resp_{n_component}.npy")
    resp = np.exp(log_resp1)
    print("resp.shape: ", resp.shape)

    data_mean = np.mean(data, axis=0)
    cluster_result = np.argmax(resp, axis=1)
    sc_idxs = np.where(cluster_result==sc)[0]
    print(f"shift {len(sc_idxs)} samples from cluster {sc} mean")
    data.iloc[sc_idxs] = data.iloc[sc_idxs] + (data_mean - data.iloc[sc_idxs].mean(axis=0))
    return data

def cluster_shift121(data, sc=0, dc=1, init_param='k-means++', random_seed=987, n_component=50):
    print(f"shift cluster {sc} to cluster {dc}")
    resp_dir = f"/nfsdata/tianhao/cluster_results/{name}/nopca/{init_param}/{random_seed}/data/"
    # resp_dir = f"../cluster_results/{name}/nopca/{init_param}/{random_seed}/data/"
    log_resp1 = np.load(resp_dir+f"class_proportions1_log_resp_{n_component}.npy")
    resp = np.exp(log_resp1)
    print("resp.shape: ", resp.shape)

    cluster_result = np.argmax(resp, axis=1)
    sc_idxs = np.where(cluster_result==sc)[0]
    print(f"shift {len(sc_idxs)} samples from cluster {sc} to cluster {dc}")
    dc_idxs = np.where(cluster_result==dc)[0]
    sc_center = np.mean(data.iloc[sc_idxs], axis=0)
    dc_center = np.mean(data.iloc[dc_idxs], axis=0)
    print(f"dc_center - sc_center\n: {(dc_center - sc_center)}")
    data.iloc[sc_idxs] = data.iloc[sc_idxs] + (dc_center - sc_center)
    return data

def cluster_shift_n21(data, sourses=[0,2,3], dc=1, init_param='k-means++', random_seed=987, n_component=50):
    print(f"shift {len(sourses)} cluster {sourses} to cluster {dc}")
    resp_dir = f"/nfsdata/tianhao/cluster_results/{name}/nopca/{init_param}/{random_seed}/data/"
    # resp_dir = f"../cluster_results/{name}/nopca/{init_param}/{random_seed}/data/"
    log_resp1 = np.load(resp_dir+f"class_proportions1_log_resp_{n_component}.npy")
    resp = np.exp(log_resp1)
    print("resp.shape: ", resp.shape)

    cluster_result = np.argmax(resp, axis=1)
    dc_idxs = np.where(cluster_result==dc)[0]
    dc_center = np.mean(data.iloc[dc_idxs], axis=0)
    for sc in sourses:
        sc_idxs = np.where(cluster_result==sc)[0]
        print(f"shift {len(sc_idxs)} samples from cluster {sc} to cluster {dc}")
        sc_center = np.mean(data.iloc[sc_idxs], axis=0)
        print(f"dc_center - sc_center\n: {(dc_center - sc_center)}")
        data.iloc[sc_idxs] = data.iloc[sc_idxs] + (dc_center - sc_center)
    return data

#for mixed data abalone
sex = data.iloc[:,0]
data = data.iloc[:,1:]
# data = laplace_noise(data)
# data = cluster_shift(data)
data = cluster_shift_n21(data, n_component=70)
data = pd.concat([sex, data], axis=1)
print("data.shape: ", data.shape)
print("data.head(): ", data.head())

filename = data_path.split('/')[-1]
noise_prefix = data_path.replace(filename, '')
file_prefix = filename.split('.')[0]
noise_data_dir = f"/nfsdata/tianhao/dataset/{noise_prefix}noise/"
# noise_data_dir = f"../dataset/{noise_prefix}noise/"
if not os.path.exists(noise_data_dir):
    print("create dir: ", noise_data_dir)
    os.makedirs(noise_data_dir)
# output_data_path = noise_data_dir + f"{file_prefix}_laplace.data"
# output_data_path = noise_data_dir + f"{file_prefix}_cluster_mean_shift.data"
output_data_path = noise_data_dir + f"{file_prefix}_cluster_321_shift_70.data"
subset1_data = pd.DataFrame(data)
subset1_data.to_csv(output_data_path, mode='w', index=False, header=True)
print(f"save data to {output_data_path}")


