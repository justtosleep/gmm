import numpy as np
import pandas as pd
import argparse

# read command line arguments
parser = argparse.ArgumentParser(description="Sample data from dataset")

# add positional arguments

# add optional arguments
parser.add_argument("--sample_num", type=int, default=1000, help="number of samples")

args = parser.parse_args()
sample_num = args.sample_num
print("sample_num: ", sample_num)

# data_path = '../dataset/covtype/covtype.data'
data_path = '../dataset/UCLAdult/UCLAdult_norm105.data'
label_path = input_dir = '../dataset/UCLAdult/UCLAdult.labels'
all_data = pd.read_csv(data_path, skipinitialspace=True).astype(np.float32)
all_label = pd.read_csv(label_path, skipinitialspace=True, header=None).astype(np.int32)
headers = list(all_data.columns.values)

np.random.seed(42)
idx = np.random.choice(len(all_data), size=sample_num , replace=False)
sample_data = all_data.iloc[idx]
sample_label = all_label.iloc[idx]

# output_data_path = '../dataset/covtype/covtype_sample50k.data'
# output_label_path = '../dataset/covtype/covtype_sample50k.labels'
output_data_path = '../dataset/UCLAdult/UCLAdult_sample1.data'
output_label_path = '../dataset/UCLAdult/UCLAdult_sample1.labels'
with open(output_data_path, 'w') as f:
    f.write(",".join(headers)+"\n")
sample_data.to_csv(output_data_path, mode='a', index=False, header=False)
np.savetxt(output_label_path, sample_label, delimiter=",", fmt="%s")

np.random.seed(40)
idx = np.random.choice(len(all_data), size=sample_num , replace=False)
sample_data = all_data.iloc[idx]
sample_label = all_label.iloc[idx]

output_data_path = '../dataset/UCLAdult/UCLAdult_sample2.data'
output_label_path = '../dataset/UCLAdult/UCLAdult_sample2.labels'
with open(output_data_path, 'w') as f:
    f.write(",".join(headers)+"\n")
sample_data.to_csv(output_data_path, mode='a', index=False, header=False)
np.savetxt(output_label_path, sample_label, delimiter=",", fmt="%s")


