import numpy as np
import pandas as pd
import os
import sys
sys.path.append("..")
from scripts.tools import create_directory

n_samples = 1000
means = list(range(1, 41))
means = list(np.array(means).reshape(-1, 2))
covs = [[[0.5, 0], [0, 0.5]] for i in range(20)]

x_range = np.linspace(-30.0, 30.0, 5)
y_range = np.linspace(-40.0, 40.0, 5)
x_coords, y_coords = np.meshgrid(x_range, y_range)
coordinates = np.column_stack((x_coords.ravel(), y_coords.ravel()))

print("means shape: ", np.array(means).shape)
print("covs shape: ", np.array(covs).shape)
print("coordinates shape: ", np.array(coordinates).shape)

for i in range(0,20):
    np.random.seed(40)
    data1 = np.random.multivariate_normal(means[i], covs[i], n_samples) + coordinates[i]
    if i == 0:
        subset1_data = data1
    else:
        subset1_data = np.vstack((subset1_data, data1))
subset1_data = random.shuffle(subset1_data)

for i in range(0,20):
    np.random.seed(42)
    data1 = np.random.multivariate_normal(means[i], covs[i], n_samples) + coordinates[i]
    if i == 0:
        subset2_data = data1
    else:
        subset2_data = np.vstack((subset2_data, data1))
subset2_data = random.shuffle(subset2_data)

print("subset1_data.shape: ", subset1_data.shape)
print("subset2_data.shape: ", subset2_data.shape)

if not os.path.exists("../dataset/toydata"):
    create_directory("../dataset/toydata")

headers = ["x", "y"]
output_data_path = "../dataset/toydata/gaussian_sample1.data"
with open(output_data_path, 'w') as f:
    f.write(",".join(headers)+"\n")
subset1_data = pd.DataFrame(subset1_data)
subset1_data.to_csv(output_data_path, mode='a', index=False, header=False)

output_data_path = "../dataset/toydata/gaussian_sample2.data"
with open(output_data_path, 'w') as f:
    f.write(",".join(headers)+"\n")
subset2_data = pd.DataFrame(subset2_data)
subset2_data.to_csv(output_data_path, mode='a', index=False, header=False)

