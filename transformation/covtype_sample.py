import numpy as np
import pandas as pd


data_path = '../dataset/covtype/covtype.data'
label_path = input_dir = '../dataset/covtype/covtype.labels'
all_data = pd.read_csv(data_path, skipinitialspace=True, header=None).astype(np.int32)
all_label = pd.read_csv(label_path, skipinitialspace=True, header=None).astype(np.int32)

np.random.seed(42)
sample_num = 50000
idx = np.random.choice(len(all_data), size=sample_num , replace=False)
sample_data = all_data.iloc[idx]
sample_label = all_label.iloc[idx]

output_data_path = '../dataset/covtype/covtype_sample50k.data'
output_label_path = '../dataset/covtype/covtype_sample50k.labels'
np.savetxt(output_data_path, sample_data, delimiter=",", fmt="%s")
np.savetxt(output_label_path, sample_label, delimiter=",", fmt="%s")

