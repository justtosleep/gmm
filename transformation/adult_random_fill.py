import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

header=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country', 'result']
data_path = "/nfsdata/tianhao/dataset/UCLAdult/UCLAdult0.data"
data = pd.read_csv(data_path, skipinitialspace=True, header=None, names=header)
print("data shape: ", data.shape)
print("data head: \n", data.head())

cat_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'result']
for col in cat_cols:
    labels = np.unique(data[col])
    print(f"col: {col}, labels: {labels}")
    labels = np.delete(labels, np.where(labels == '?'))
    data[col].replace('?', labels[random.randint(0, len(labels)-1)], inplace=True)
    labels = np.unique(data[col])
    print(f"After impute, labels: {labels}")
    # data[col] = label_encoder.fit_transform(data[col])
print("data shape", data.shape)
print("data head", data.head())
cols = data.columns
print("cols: ", cols)


output_path = '/nfsdata/tianhao/dataset/UCLAdult/UCLAdult.data'
data.to_csv(output_path, mode='a', index=False)
print("Save data to", output_path)

data = pd.read_csv(output_path, skipinitialspace=True)
print("data shape", data.shape)
print("data head", data.head())
