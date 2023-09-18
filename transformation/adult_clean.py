import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import chi2_contingency


# 1. Read data
print("Start reading data")
names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital loss','hours-per-week','native-country', 'result']
main_data = pd.read_csv('../dataset/UCLAdult/UCLAdult.data', names = names, skipinitialspace=True).fillna('?')
main_data.drop(index=main_data.index[0],axis=0,inplace=True)
y = np.array(main_data['result'])
main_data['result'] = np.array([0 if ic == "<=50K" else 1 for ic in y])

# 2. Remove missing values
print("Start removing missing values")
main_data['workclass'].value_counts()
main_data['workclass'].loc[main_data['workclass'].str.contains('\?')]='Private'
main_data['occupation'].value_counts()
main_data['occupation'].loc[main_data['occupation'].str.contains('\?')]='Prof-specialty'
main_data['native-country'].value_counts()
main_data['native-country'].loc[main_data['native-country'].str.contains('\?')]='United-States'
print("All missing values are removed")

# 3. Convert categorical data to numerical data
print("Start converting categorical data to numerical data")
main_data['fnlwgt'] = pd.to_numeric(main_data['fnlwgt'], errors='coerce')
main_data['education-num'] = pd.to_numeric(main_data['education-num'], errors='coerce')
main_data['capital-gain'] = pd.to_numeric(main_data['capital-gain'], errors='coerce')
main_data['capital loss'] = pd.to_numeric(main_data['capital loss'], errors='coerce')
main_data['hours-per-week'] = pd.to_numeric(main_data['hours-per-week'], errors='coerce')
# boxplot = main_data.boxplot(column=['fnlwgt'],figsize=(15, 10),boxprops=dict(color='red'))
# main_data.hist(bins=30, figsize=(15, 10))

# 4. Detect outliers
print("Start detecting outliers")
detection_outlier(main_data['fnlwgt'])
detection_outlier(main_data['capital-gain'])
detection_outlier(main_data['capital loss'])
detection_outlier(main_data['hours-per-week'])
detection_outlier(main_data['education-num'])

# 5. Remove duplicates
print("Start removing duplicates")
main_data.drop_duplicates()
duplicate = main_data[main_data.duplicated()]
pd.set_option("display.max_rows", None)

# 6. Correlation matrix
# categorical_main_data=main_data[['workclass','education','marital-status','occupation','relationship','race','sex','native-country']]
# numerical_main_data=main_data[['fnlwgt','capital-gain','capital loss','hours-per-week']]
# numerical_main_data['age']=pd.to_numeric(main_data['age'])
# #Correlation matrix with colorful indicators
# f = plt.figure(figsize=(19, 15))
# plt.matshow(numerical_main_data.corr(), fignum=f.number)
# plt.xticks(range(numerical_main_data.select_dtypes(['number']).shape[1]), numerical_main_data.select_dtypes(['number']).columns, fontsize=14, rotation=45)
# plt.yticks(range(numerical_main_data.select_dtypes(['number']).shape[1]), numerical_main_data.select_dtypes(['number']).columns, fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16);
# corr = numerical_main_data.corr()
# corr.style.background_gradient(cmap='coolwarm')

# 7. Chi-square test
# contigency= pd.crosstab(categorical_main_data['workclass'], categorical_main_data['occupation'])
# contigency_pct = pd.crosstab(categorical_main_data['workclass'], categorical_main_data['occupation'], normalize='index')
# plt.figure(figsize=(12,8))
# sns.heatmap(contigency, annot=True, cmap="YlGnBu")
# c, p, dof, expected = chi2_contingency(contigency)
# print('Chi2: \n',c)
# print('The p-value of the test:\n',p)

# 8. Save cleaned data
print("Start saving cleaned data")
output_best_path = '../dataset/UCLAdult/UCLAdult_clean.data'
with open(output_best_path, 'w') as f:
    main_data.to_csv(f, index=False)

print("Data cleaning is done")