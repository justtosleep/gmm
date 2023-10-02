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
parser.add_argument("--sample_num", type=int, default=1000, help="number of samples")

args = parser.parse_args()
domain = args.dataset
data_path = args.data_path
method = args.method
sample_num = args.sample_num
print("domain: ", domain)
print("data_path: ", data_path)
print("method: ", method)
print("sample_num: ", sample_num)

# data_path = '../dataset/covtype/covtype.data'
name = domain+"_"+method if method != "" else domain
input_path = "../dataset/" + data_path
label_path = "../dataset/" + domain + "/" + domain + ".labels"
all_data = pd.read_csv(input_path, skipinitialspace=True).astype(np.float32)
all_label = pd.read_csv(label_path, skipinitialspace=True, header=None).astype(np.int32)
headers = list(all_data.columns.values)

np.random.seed(42)
idx = np.random.choice(len(all_data), size=sample_num , replace=False)
sample_data = all_data.iloc[idx]
sample_label = all_label.iloc[idx]

# output_data_path = '../dataset/covtype/covtype_sample50k.data'
# output_label_path = '../dataset/covtype/covtype_sample50k.labels'
output_data_path = "../dataset/" + data_path.split('.')[0]+"_sample1.data"
output_label_path = "../dataset/" + data_path.split('.')[0]+"_sample1.labels"
with open(output_data_path, 'w') as f:
    f.write(",".join(headers)+"\n")
sample_data.to_csv(output_data_path, mode='a', index=False, header=False)
np.savetxt(output_label_path, sample_label, delimiter=",", fmt="%s")

np.random.seed(40)
idx = np.random.choice(len(all_data), size=sample_num , replace=False)
sample_data = all_data.iloc[idx]
sample_label = all_label.iloc[idx]

output_data_path = '../dataset/'+data_path.split('.')[0]+'_sample2.data'
output_label_path = '../dataset/'+data_path.split('.')[0]+'_sample2.labels'
with open(output_data_path, 'w') as f:
    f.write(",".join(headers)+"\n")
sample_data.to_csv(output_data_path, mode='a', index=False, header=False)
np.savetxt(output_label_path, sample_label, delimiter=",", fmt="%s")


