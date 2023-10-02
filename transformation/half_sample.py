import numpy as np
import pandas as pd
import argparse

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
np.random.seed(42)

name = domain+"_"+method if method != "" else domain
input_path = "../dataset/" + data_path
data = pd.read_csv(input_path, skipinitialspace=True).astype(np.int32)
headers = list(data.columns.values)
idx = np.random.choice(data.shape[0], data.shape[0]//2, replace=False)
subset1_data = data.iloc[idx]
subset2_data = data.drop(idx)
print("subset1_data.shape: ", subset1_data.shape)
print("subset2_data.shape: ", subset2_data.shape)

output_data_path = "../dataset/" + data_path.split('.')[0]+"_sample1.data"
with open(output_data_path, 'w') as f:
    f.write(",".join(headers)+"\n")
subset1_data = pd.DataFrame(subset1_data)
subset1_data.to_csv(output_data_path, mode='a', index=False, header=False)

output_data_path = '../dataset/'+data_path.split('.')[0]+'_sample2.data'
with open(output_data_path, 'w') as f:
    f.write(",".join(headers)+"\n")
subset2_data = pd.DataFrame(subset2_data)
subset2_data.to_csv(output_data_path, mode='a', index=False, header=False)

# #for labels
# name = domain+"_"+method if method != "" else domain
# input_path = "../dataset/" + data_path
# label_path = "../dataset/" + domain + "/" + domain + ".labels"
# data = pd.read_csv(input_path, skipinitialspace=True).astype(np.float32)
# labels = pd.read_csv(label_path, skipinitialspace=True, header=None).astype(np.int32)
# headers = list(data.columns.values)

# subset1_data = []
# subset1_label = []
# subset2_data = []
# subset2_label = []
# unique_labels = np.unique(labels)
# for label in unique_labels:
#     idx = np.where(labels == label)[0]
#     np.random.shuffle(idx)

#     half_len = len(idx) // 2
#     subset1_data.append(data.iloc[idx[:half_len]])
#     subset1_label.append([label]*subset1_data[-1].shape[0])
#     subset2_data.append(data.iloc[idx[half_len:]])
#     subset2_label.append([label]*subset2_data[-1].shape[0])

# subset1_data = np.concatenate(subset1_data, axis=0)
# subset1_label = np.concatenate(subset1_label, axis=0)
# subset1 = list(zip(subset1_data, subset1_label))
# np.random.shuffle(subset1)
# subset1_data, subset1_label = zip(*subset1)
# subset1_data = np.vstack(subset1_data)

# subset2_data = np.concatenate(subset2_data, axis=0)
# subset2_label = np.concatenate(subset2_label, axis=0)
# subset2 = list(zip(subset2_data, subset2_label))
# np.random.shuffle(subset2)
# subset2_data, subset2_label = zip(*subset2)
# subset2_data = np.vstack(subset2_data)
# print("subset1_data.shape: ", subset1_data.shape)
# print("subset2_data.shape: ", subset2_data.shape)

# output_data_path = "../dataset/" + data_path.split('.')[0]+"_sample1.data"
# output_label_path = "../dataset/" + data_path.split('.')[0]+"_sample1.labels"
# with open(output_data_path, 'w') as f:
#     f.write(",".join(headers)+"\n")
# subset1_data = pd.DataFrame(subset1_data)
# subset1_data.to_csv(output_data_path, mode='a', index=False, header=False)
# np.savetxt(output_label_path, subset1_label, delimiter=",", fmt="%d")

# output_data_path = '../dataset/'+data_path.split('.')[0]+'_sample2.data'
# output_label_path = '../dataset/'+data_path.split('.')[0]+'_sample2.labels'
# with open(output_data_path, 'w') as f:
#     f.write(",".join(headers)+"\n")
# subset2_data = pd.DataFrame(subset2_data)
# subset2_data.to_csv(output_data_path, mode='a', index=False, header=False)
# np.savetxt(output_label_path, subset2_label, delimiter=",", fmt="%d")


