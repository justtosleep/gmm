import pandas as pd
from sklearn.preprocessing import LabelEncoder

#read data
data_file = "../dataset/abalone/abalone.data"  
data = pd.read_csv(data_file, header=None, float_precision='round_trip')

#extract target column Sex and encode it
target_col = data.iloc[:, 0]
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(target_col)
encoded_labels = pd.DataFrame(encoded_labels, columns=['label'])
# print("LabelEncoder encoding result：", encoded_labels)

#merge the target column and the remaining columns
remaining_cols = data.iloc[:, 1:]  
merged_data = pd.concat([encoded_labels, remaining_cols], axis=1)

#save the merged data
output_file = "../dataset/abalone/abalone_1hot.data"
merged_data.to_csv(output_file, index=False, header=False)

print("One-hot encoding finished!")
