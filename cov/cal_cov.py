import pandas as pd
import numpy as np
import os
import sys
import argparse
from dython.nominal import associations, identify_nominal_columns

# read command line arguments
parser = argparse.ArgumentParser(description="Calculate the correlation of dataset.")

# add positional arguments
# parser.add_argument("dataset", type=str, help="name of dataset")
parser.add_argument("path", type=str, help="path of dataset")

#add optional arguments
parser.add_argument("--colstype", type=str, default="", help="type of dataset cols to calculate correlation")

args = parser.parse_args()
input_path = args.path
cols_type = args.colstype

#1. Read data
domain = input_path.split('/')[-1].split('.')[0]
print("domain: ", domain)
# data = pd.read_csv(input_path, skipinitialspace=True)

if domain == 'abalone':
    features = [
    'Sex',
    'Length',
    'Diameter',
    'Height',
    'Whole weight',
    'Shucked weight',
    'Viscera weight',
    'Shell weight',
    'Rings'
    ]
    num_cols = features[1:]
    cat_cols = [cols for cols in features if cols not in num_cols]
    data = pd.read_csv(input_path, skipinitialspace=True, names=features)
elif domain == 'covtype_restored':
    data = pd.read_csv(input_path, skipinitialspace=True)
    features = list(data.columns)
    cat_cols = [cols for cols in features if cols not in num_cols]
    num_cols = features[:-1]
elif domain == 'adult':
    features = [
    'age',
    'workclass',
    'fnlwgt',
    'education',
    'education-num',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'capital-gain',
    'capital loss',
    'hours-per-week',
    'native-country',
    'result'
    ]
    num_cols = [
    'age',
    'fnlwgt',
    'education-num',
    'capital-gain',
    'capital loss',
    'hours-per-week',
    ]
    cat_cols = [cols for cols in features if cols not in num_cols]
    data = pd.read_csv(input_path, skipinitialspace=True, names=features)
elif domain == 'arizona':
    features = [
    "SPLIT", "URBAN", "VETWWI", "SEX", "RESPONDT", "SLREC", "LABFORCE", "SSENROLL", "SCHOOL", "FARM", "OWNERSHP",
    "SPANNAME", "METRO", "EMPSTAT", "HISPAN", "CITIZEN", "NATIVITY", "NCHLT5", "WARD", "RACE", "MARST", "WKSWORK2",
    "GQ", "VET1940", "UCLASSWK", "SIZEPL", "HRSWORK2", "VETPER", "HISPRULE", "MIGRATE5", "CLASSWKR", "GQTYPE",
    "VETCHILD", "VETSTAT", "SAMESEA5", "SAMEPLAC", "MIGTYPE5", "INCNONWG", "MARRNO", "SEA", "OWNERSHPD", "FAMSIZE",
    "EMPSTATD", "WKSWORK1", "OCCSCORE", "AGEMARR", "MIGRATE5D", "MTONGUE", "SEI", "HRSWORK1", "CLASSWKRD", "HIGRADE",
    "CHBORN", "EDUC", "VETSTATD", "AGEMONTH", "GQFUNDS", "AGE", "SUPDIST", "COUNTY", "HISPAND", "METAREA", "CITYPOP",
    "URBPOP", "RACED", "PRESGL", "MBPL", "BPL", "FBPL", "IND1950", "MIGSEA5", "OCC", "OCC1950", "IND", "MIGPLAC5",
    "DURUNEMP", "GQTYPED", "UOCC", "EDUCD", "UOCC95", "UIND", "HIGRADED", "ENUMDIST", "METAREAD", "MTONGUED",
    "EDSCOR50", "MIGMET5", "RENT", "CITY", "NPBOSS50", "MIGCOUNTY", "MIGCITY5", "ERSCOR50", "BPLD", "FBPLD",
    "MBPLD", "INCWAGE", "VALUEH"
    ]
    num_cols = [
    "SPLIT", "RESPONDT", "SLREC", "LABFORCE", "SSENROLL", "WKSWORK2", "NCHLT5", "WKSWORK1",
    "OCCSCORE", "AGEMARR", "MIGRATE5D", "SEI", "HRSWORK1", "HRSWORK2", "DURUNEMP", "ERSCOR50",
    "EDSCOR50", "NPBOSS50", "RENT", "VALUEH"
    ]
    cat_cols = [
    "URBAN", "VETWWI", "SEX", "SCHOOL", "FARM", "OWNERSHP", "SPANNAME", "METRO", "EMPSTAT", "HISPAN",
    "CITIZEN", "NATIVITY", "WARD", "RACE", "MARST", "GQ", "VET1940", "UCLASSWK", "SIZEPL", "VETPER",
    "HISPRULE", "MIGRATE5", "CLASSWKR", "GQTYPE", "VETCHILD", "VETSTAT", "SAMESEA5", "SAMEPLAC",
    "MIGTYPE5", "INCNONWG", "MARRNO", "SEA", "OWNERSHPD", "FAMSIZE", "EMPSTATD", "WKSWORK1", "OCC",
    "EDUCD", "UOCC", "UOCC95", "UIND", "HIGRADED", "ENUMDIST", "METAREAD", "MTONGUED", "MIGCITY5",
    "MIGCOUNTY", "CITY", "FBPLD", "MBPLD", "BPLD"
    ]
    data = pd.read_csv(input_path, skipinitialspace=True)
    print("data columns: ", data.columns)
elif domain == 'vermont':
    features = [
    "METAREA", "METAREAD", "SPLIT", "SCHOOL", "SEX", "RESPONDT", "SLREC", "LABFORCE", "SSENROLL", "FARM",
    "OWNERSHP", "VETWWI", "URBAN", "SPANNAME", "EMPSTAT", "CITIZEN", "HISPAN", "RACE", "NATIVITY", "NCHLT5",
    "WKSWORK2", "GQ", "MARST", "SIZEPL", "HRSWORK2", "HISPRULE", "VETPER", "VET1940", "UCLASSWK", "CLASSWKR",
    "GQTYPE", "VETSTAT", "SAMESEA5", "SAMEPLAC", "MIGTYPE5", "VETCHILD", "MIGRATE5", "MARRNO", "INCNONWG",
    "WARD", "OWNERSHPD", "FAMSIZE", "EMPSTATD", "WKSWORK1", "MTONGUE", "AGEMARR", "OCCSCORE", "MIGRATE5D",
    "SEI", "HRSWORK1", "CLASSWKRD", "CHBORN", "EDUC", "AGEMONTH", "GQFUNDS", "VETSTATD", "HIGRADE", "AGE",
    "SUPDIST", "COUNTY", "CITYPOP", "URBPOP", "RACED", "SEA", "HISPAND", "ENUMDIST", "PRESGL", "MBPL", "FBPL",
    "BPL", "MIGSEA5", "IND1950", "OCC", "DURUNEMP", "HIGRADED", "UOCC95", "OCC1950", "UOCC", "MIGPLAC5",
    "IND", "UIND", "EDUCD", "GQTYPED", "MTONGUED", "MIGCITY5", "ERSCOR50", "CITY", "EDSCOR50", "MIGMET5",
    "MIGCOUNTY", "RENT", "NPBOSS50", "MBPLD", "FBPLD", "BPLD", "INCWAGE", "VALUEH"
    ]
    num_cols = [
        "METAREA", "METAREAD", "SPLIT", "SCHOOL", "SLREC", "VETWWI", "URBAN", "EMPSTAT", "NATIVITY", "NCHLT5", 
        "WKSWORK2", "GQ", "MARST", "SIZEPL", "HRSWORK2", "HISPRULE", "VETPER", "VET1940", "UCLASSWK", "CLASSWKR", 
        "GQTYPE", "VETSTAT", "SAMESEA5", "SAMEPLAC", "MIGTYPE5", "VETCHILD", "MARRNO", "INCNONWG", "WARD", 
        "OWNERSHPD", "FAMSIZE", "EMPSTATD", "WKSWORK1", "AGEMARR", "OCCSCORE", "MIGRATE5D", "SEI", "HRSWORK1", 
        "CLASSWKRD", "CHBORN", "EDUC", "AGEMONTH", "GQFUNDS", "VETSTATD", "HIGRADE", "AGE", "SUPDIST", "CITYPOP", 
        "URBPOP", "SEA", "ENUMDIST", "PRESGL", "MBPL", "FBPL", "BPL", "MIGSEA5", "IND1950", "DURUNEMP", 
        "HIGRADED", "UOCC95", "OCC1950", "UOCC", "MIGPLAC5", "IND", "UIND", "EDUCD", "GQTYPED", "ERSCOR50", 
        "CITY", "EDSCOR50", "MIGMET5", "MIGCOUNTY", "RENT", "NPBOSS50", "MBPLD", "FBPLD", "BPLD", "INCWAGE", "VALUEH"
    ]
    cat_cols = [
        "SEX", "RESPONDT", "OWNERSHP", "FARM", "CITIZEN", "HISPAN", "RACE", "GQ", "NATIVITY", "GQTYPE", 
        "VETSTAT", "SAMESEA5", "SAMEPLAC", "MIGTYPE5", "VETCHILD", "OWNERSHPD", "FAMSIZE", "EMPSTATD", "SEA", 
        "MTONGUED", "VETSTATD", "HIGRADE", "SUPDIST", "COUNTY", "RACED", "HISPAND", "ENUMDIST", "BPL", 
        "FBPL", "MBPL", "IND1950", "OCC", "UOCC95", "OCC1950", "UOCC", "MIGPLAC5", "IND", "UIND", "EDUCD", 
        "GQTYPED", "MTONGUED", "CITY"
    ]
    data = pd.read_csv(input_path, skipinitialspace=True)
   


# col = data.iloc[:, -2]
# data.iloc[:, -2] = -col

# print("data shape: ", data.shape)
# categorical_columns = identify_nominal_columns(data)
# print("categorical_columns: ", categorical_columns)
# print("categorical_columns len: ", len(categorical_columns))
# categorical_cols = data.columns
# print("categorical_cols: ", categorical_cols)
# print("categorical_cols len: ", len(categorical_cols))

if cols_type == 'num':
    data = data[num_cols].astype(np.int32)
    print("data num shape: ", data.shape)
    print("data num head: ", data[:5])
elif cols_type == 'cat':
    data = data[cat_cols]
    print("data cat shape: ", data.shape)
    print("data cat head: ", data[:5])

width = data.shape[1]
if data.shape[1] > 10:
    width = data.shape[1] / 2
height = width
# complete_correlation= associations(data, filename= '{}_correlation.png'.format(domain), figsize=(width,height), nominal_columns=list(categorical_cols))
out_path = './result/{}_correlation_{}.png'.format(domain, cols_type) if cols_type != '' else './result/{}_correlation.png'.format(domain)
complete_correlation= associations(data, filename=out_path, figsize=(width,height), numerical_columns=num_cols)
print("complete_correlation: ", complete_correlation)
df_complete_corr=complete_correlation['corr']
df_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None)

# associations(data,                      # 数据集，一个 DataFrame 或类似的数据结构
#              nominal_columns='auto',    # 指定分类变量的列，可以是列名的列表或 'auto'（自动检测）
#              numerical_columns=None,    # 指定数值变量的列，可以是列名的列表或 None
#              mark_columns=False,        # 是否标记与目标变量的相关性（仅用于分类目标变量）
#              nom_nom_assoc='cramer',    # 分类-分类变量关联性的度量方法（默认是 Cramer's V）
#              num_num_assoc='pearson',   # 数值-数值变量关联性的度量方法（默认是 Pearson 相关系数）
#              bias_correction=True,      # 是否进行偏差校正以改善关联性估计
#              nan_strategy=_REPLACE,     # 处理缺失值的策略，默认替换为特殊值
#              nan_replace_value=_DEFAULT_REPLACE_VALUE,  # 替换缺失值的默认值
#              ax=None,                   # Matplotlib Axes 对象，用于绘制关联性图
#              figsize=None,              # 图的大小
#              annot=True,                # 是否在图中显示数字注释
#              fmt='.2f',                 # 数字注释的格式
#              cmap=None,                 # 色彩映射（colormap）
#              sv_color='silver',         # 关联性图的变量标签颜色
#              cbar=True,                 # 是否显示颜色条
#              vmax=1.0,                  # 颜色条的最大值
#              vmin=None,                 # 颜色条的最小值
#              plot=True,                 # 是否绘制关联性图
#              compute_only=False,        # 是否仅计算关联性，不绘制图
#              clustering=False,          # 是否进行聚类
#              title=None,                # 图的标题
#              filename=None              # 保存图的文件名
#              )             
