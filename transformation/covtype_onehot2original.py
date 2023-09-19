import pandas as pd


# 定义One-Hot编码的列名
quantitative_columns = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                        'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                        'Horizontal_Distance_To_Fire_Points']
one_hot_columns = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                   'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7',
                   'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
                   'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 'Soil_Type19',
                   'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
                   'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
                   'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
                   'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

column_mapping = {
    'Wilderness_Area1': 'Rawah Wilderness Area',
    'Wilderness_Area2': 'Neota Wilderness Area',
    'Wilderness_Area3': 'Comanche Peak Wilderness Area',
    'Wilderness_Area4': 'Cache la Poudre Wilderness Area',
    'Soil_Type1': 'Cathedral family - Rock outcrop complex, extremely stony',
    'Soil_Type2': 'Vanet - Ratake families complex, very stony',
    'Soil_Type3': 'Haploborolis - Rock outcrop complex, rubbly',
    'Soil_Type4': 'Ratake family - Rock outcrop complex, rubbly',
    'Soil_Type5': 'Vanet family - Rock outcrop complex complex, rubbly',
    'Soil_Type6': 'Vanet - Wetmore families - Rock outcrop complex, stony',
    'Soil_Type7': 'Gothic family',
    'Soil_Type8': 'Supervisor - Limber families complex',
    'Soil_Type9': 'Troutville family, very stony',
    'Soil_Type10': 'Bullwark - Catamount families - Rock outcrop complex, rubbly',
    'Soil_Type11': 'Bullwark - Catamount families - Rock land complex, rubbly',
    'Soil_Type12': 'Legault family - Rock land complex, stony',
    'Soil_Type13': 'Catamount family - Rock land - Bullwark family complex, rubbly',
    'Soil_Type14': 'Pachic Argiborolis - Aquolis complex',
    'Soil_Type15': 'unspecified in the USFS Soil and ELU Survey',
    'Soil_Type16': 'Cryaquolis - Cryoborolis complex',
    'Soil_Type17': 'Gateview family - Cryaquolis complex',
    'Soil_Type18': 'Rogert family, very stony',
    'Soil_Type19': 'Typic Cryaquolis - Borohemists complex',
    'Soil_Type20': 'Typic Cryaquepts - Typic Cryaquolls complex',
    'Soil_Type21': 'Typic Cryaquolls - Leighcan family, till substratum complex',
    'Soil_Type22': 'Leighcan family, till substratum, extremely bouldery',
    'Soil_Type23': 'Leighcan family, till substratum - Typic Cryaquolls complex',
    'Soil_Type24': 'Leighcan family, extremely stony',
    'Soil_Type25': 'Leighcan family, warm, extremely stony',
    'Soil_Type26': 'Granile - Catamount families complex, very stony',
    'Soil_Type27': 'Leighcan family, warm - Rock outcrop complex, extremely stony',
    'Soil_Type28': 'Leighcan family - Rock outcrop complex, extremely stony',
    'Soil_Type29': 'Como - Legault families complex, extremely stony',
    'Soil_Type30': 'Como family - Rock land - Legault family complex, extremely stony',
    'Soil_Type31': 'Leighcan - Catamount families complex, extremely stony',
    'Soil_Type32': 'Catamount family - Rock outcrop - Leighcan family complex, extremely stony',
    'Soil_Type33': 'Leighcan - Catamount families - Rock outcrop complex, extremely stony',
    'Soil_Type34': 'Cryorthents - Rock land complex, extremely stony',
    'Soil_Type35': 'Cryumbrepts - Rock outcrop - Cryaquepts complex',
    'Soil_Type36': 'Bross family - Rock land - Cryumbrepts complex, extremely stony',
    'Soil_Type37': 'Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony',
    'Soil_Type38': 'Leighcan - Moran families - Cryaquolls complex, extremely stony',
    'Soil_Type39': 'Moran family - Cryorthents - Leighcan family complex, extremely stony',
    'Soil_Type40': 'Moran family - Cryorthents - Rock land complex, extremely stony'
}

df = pd.read_csv('../dataset/covtype/covtype.data', 
                 header=None, 
                 skipinitialspace=True, 
                 names=quantitative_columns + one_hot_columns + ['Cover_Type']
                 )
print("df.head(): \n", df.head())

restored_df = df.copy()

restored_df['Soil_Type'] = None

# 遍历字典，将One-Hot编码的列还原为新列的值
for column, cat_name in column_mapping.items():
    restored_df['Soil_Type'] = restored_df.apply(lambda row: cat_name if row[column] == 1 else row['Soil_Type'], axis=1)

# 删除原始的One-Hot编码列
restored_df = restored_df.drop(column_mapping.keys(), axis=1)
print("restored_df.head(): \n", restored_df.head())

# 保存结果
restored_df.to_csv('../dataset/covtype/covtype_restored.csv', index=False)



