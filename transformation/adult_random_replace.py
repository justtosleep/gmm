import pandas as pd
import numpy as np
import random
import os


names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country']
numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital loss', 'hours-per-week']
cat_features = [col for col in names if col not in numerical_features]

data_path = "/nfsdata/tianhao/dataset/UCLAdult/UCLAdult.data"
data = pd.read_csv(data_path, skipinitialspace=True)
print("data.shape", data.shape)
print("data.head\n", data.head())

cols = data.columns
print("cols: ", cols.values.tolist())

save_dir = "/nfsdata/tianhao/dataset/UCLAdult/noise/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

random.seed(42)
replace_cols = len(cat_features)
replace_probability = 0.2
for num in range(replace_cols):
    col = cat_features[num]
    print(f"col: {col}")
    unique_value = list(set(data[col].values.tolist()))
    print("len(unique_value)", len(unique_value))
    for i in range(data.shape[0]):
        if random.random() < replace_probability:
            while True:
                random_value = random.choice(unique_value)
                if random_value != data.loc[i, col]:
                    data.loc[i, col] = random_value
                    break

    print(f"*****Replace {num+1} cols*****")
    save_path = save_dir+f"UCLAdult_replace_{num+1}.csv"
    print("data.shape", data.shape)
    print("data.head\n", data.head())
    with open(save_path, 'w') as f:
        data.to_csv(f, index=False)
    print("Save to ", save_path)

    data_validate = pd.read_csv(save_path, skipinitialspace=True)
    print("data_validate.shape", data_validate.shape)
    print("data_validate.head\n", data_validate.head())
