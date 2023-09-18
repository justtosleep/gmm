from prince import FAMD
import pandas as pd
import numpy as np
# from dp.py import laplace_mech, pct_error

# 1. Read data
print("Start reading data")
input_path = '../dataset/UCLAdult/UCLAdult.data'
names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country', 'result']
original_data = pd.read_csv(input_path, names = names, skipinitialspace=True, header=None)
print("original_data.shape", original_data.shape)


# 2. Preprocess data
income = np.array(original_data['result'])
original_data['result'] = np.array([0 if ic == "<=50K" else 1 for ic in income])

# 标记定性变量
qualitative_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'result']
quantitative_data = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital loss', 'hours-per-week']

# 将定性变量列的数据类型设置为 category
original_data[qualitative_vars] = original_data[qualitative_vars].astype('category')
original_data[quantitative_data] = original_data[quantitative_data].astype('float64')

X_train = original_data.iloc[:, 0:-1]
print("X_train.head", X_train.head())
print("X_train.shape", X_train.shape)

# 3. Apply FAMD
print("Start applying FAMD")
famd = FAMD(n_components = 100, random_state = 101).fit(X_train)
famdX = famd.transform(X_train)

# # 3. elbow method
# # Calculate eigenvalues
# eigenvalues = famd.eigenvalues_

# # Calculate explained variance ratio
# explained_variance_ratio = eigenvalues / np.sum(eigenvalues)
# cumulative_explained_variance_ratio = np.cumsum(explained_variance_ratio)

# # Plot the cumulative explained variance ratio
# plt.plot(range(1, len(cumulative_explained_variance_ratio) + 1), cumulative_explained_variance_ratio, marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance Ratio')
# plt.title('Cumulative Explained Variance Ratio by Components')
# plt.grid()
# plt.show()

# 5. Save the result
output_path = '../dataset/UCLAdult/UCLAdult_famd.data'
data = pd.DataFrame(famdX)
print("data.shape", data.shape)
with open(output_path, 'w') as f:
    data.to_csv(f, index=False, header=False)

print("FAMD finished")