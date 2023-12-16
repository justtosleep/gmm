import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce



def custom_scaler(data, old_min, old_max, new_min=0, new_max=1):
    # old_min = np.min(data)
    # old_max = np.max(data)
    return (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min

def norm(data, num_cols, cat_cols, encoder, save_path):
    cols = data.columns.values.tolist()
    valid_num_cols = [col for col in cols if col in num_cols]
    valid_cat_cols = [col for col in cols if col in cat_cols]
    print("valid_num_cols: ", valid_num_cols)
    print("valid_cat_cols: ", valid_cat_cols)
    num_data = data[valid_num_cols]
    print("num_data.shape: ", num_data.shape)
    for col in valid_num_cols:
        col_min = num_min[col]
        col_max = num_max[col]
        num_data.loc[:, col] = custom_scaler(num_data[col], col_min, col_max)
    print("arter norm num_data.shape: ", num_data.shape)

    cat_data = data[valid_cat_cols].astype(str)
    print("cat_data.shape: ", cat_data.shape)
    one_hot_data = encoder.transform(cat_data)
    one_hot_data = pd.DataFrame(one_hot_data)
    print("arter norm cat_data.shape: ", one_hot_data.shape)

    new_data = pd.concat([num_data, one_hot_data], axis=1)
    print("new_data.shape: ", new_data.shape)
    print("new_data.head\n", new_data.head())
    new_data.to_csv(save_path, index=False)

def split_cols(cols_str, cols_type_str):
    cols = cols_str.split(",")
    cols_type = cols_type_str.split(",")
    print("len(cols): ", len(cols))
    print("len(cols_type): ", len(cols_type))

    num_cols = [cols[i] for i in range(len(cols)) if cols_type[i] == "I"]
    cat_cols = [cols[i] for i in range(len(cols)) if cols_type[i] == "C"]
    print("num_cols: ", num_cols)
    print("cat_cols: ", cat_cols)
    return num_cols, cat_cols

vm_cols_str = "METAREA,METAREAD,SPLIT,METRO,SCHOOL,SEX,RESPONDT,SLREC,LABFORCE,SSENROLL,FARM,OWNERSHP,VETWWI,URBAN,SPANNAME,EMPSTAT,CITIZEN,HISPAN,RACE,NATIVITY,NCHLT5,WKSWORK2,GQ,MARST,SIZEPL,HRSWORK2,HISPRULE,VETPER,VET1940,UCLASSWK,CLASSWKR,GQTYPE,VETSTAT,SAMESEA5,SAMEPLAC,MIGTYPE5,VETCHILD,MIGRATE5,MARRNO,INCNONWG,WARD,OWNERSHPD,FAMSIZE,EMPSTATD,WKSWORK1,MTONGUE,AGEMARR,OCCSCORE,MIGRATE5D,SEI,HRSWORK1,CLASSWKRD,CHBORN,EDUC,AGEMONTH,GQFUNDS,VETSTATD,HIGRADE,AGE,SUPDIST,COUNTY,CITYPOP,URBPOP,RACED,SEA,HISPAND,ENUMDIST,PRESGL,MBPL,FBPL,BPL,MIGSEA5,IND1950,OCC,DURUNEMP,HIGRADED,UOCC95,OCC1950,UOCC,MIGPLAC5,IND,UIND,EDUCD,GQTYPED,MTONGUED,MIGCITY5,ERSCOR50,CITY,EDSCOR50,MIGMET5,MIGCOUNTY,RENT,NPBOSS50,MBPLD,FBPLD,BPLD,VALUEH,INCWAGE"
vm_cols_type_str = "C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,I,I,C,C,I,I,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,I,C,I,C,I,I,C,I,I,C,I,I,I,C,C,I,I,I,C,I,I,C,C,C,C,I,C,C,C,C,C,C,I,C,C,C,C,C,C,C,I,C,C,C,I,C,I,C,C,I,I,C,C,C,I,C"
vm_num_cols = ['NCHLT5', 'SIZEPL', 'FAMSIZE', 'EDUC', 'HIGRADE', 'AGE', 'SUPDIST', 'URBPOP', 'EDUCD']
vm_cat_cols = ['METAREAD', 'SPLIT', 'METRO', 'SCHOOL', 'SEX', 'RESPONDT', 'SLREC', 'LABFORCE', 'FARM', 'OWNERSHP', 'URBAN', 'SPANNAME', 'EMPSTAT', 'HISPAN', 'RACE', 'GQ', 'MARST', 'HISPRULE', 'SAMESEA5', 'SAMEPLAC', 'MIGTYPE5', 'MIGRATE5', 'MARRNO', 'INCNONWG', 'OWNERSHPD', 'EMPSTATD', 'MIGRATE5D', 'CLASSWKRD', 'VETSTATD', 'COUNTY', 'RACED', 'SEA', 'HISPAND', 'ENUMDIST', 'BPL', 'MIGSEA5', 'OCC', 'HIGRADED', 'MIGPLAC5', 'IND', 'GQTYPED', 'MTONGUED', 'MBPLD', 'FBPLD', 'BPLD', 'INCWAGE']
vm_num_min = {
    "NCHLT5": 0,
    "SIZEPL": 1,
    "FAMSIZE": 0,
    "EDUC": 0,
    "HIGRADE": 1,
    "AGE": 0,
    "SUPDIST": 0,
    "URBPOP": 0,
    "EDUCD": 0
}
vm_num_max = {
    "NCHLT5": 9,
    "SIZEPL": 90,
    "FAMSIZE": 29,
    "EDUC": 11,
    "HIGRADE": 99,
    "AGE": 135,
    "SUPDIST": 520,
    "URBPOP": 276,
    "EDUCD": 116
}

ar_cols_str = "SPLIT,URBAN,VETWWI,SEX,RESPONDT,SLREC,LABFORCE,SSENROLL,SCHOOL,FARM,OWNERSHP,SPANNAME,METRO,EMPSTAT,HISPAN,CITIZEN,NATIVITY,NCHLT5,WARD,RACE,MARST,WKSWORK2,GQ,VET1940,UCLASSWK,SIZEPL,HRSWORK2,VETPER,HISPRULE,MIGRATE5,CLASSWKR,GQTYPE,VETCHILD,VETSTAT,SAMESEA5,SAMEPLAC,MIGTYPE5,INCNONWG,MARRNO,SEA,OWNERSHPD,FAMSIZE,EMPSTATD,WKSWORK1,OCCSCORE,AGEMARR,MIGRATE5D,MTONGUE,SEI,HRSWORK1,CLASSWKRD,HIGRADE,CHBORN,EDUC,VETSTATD,AGEMONTH,GQFUNDS,AGE,SUPDIST,COUNTY,HISPAND,METAREA,CITYPOP,URBPOP,RACED,PRESGL,MBPL,BPL,FBPL,IND1950,MIGSEA5,OCC,OCC1950,IND,MIGPLAC5,DURUNEMP,GQTYPED,UOCC,EDUCD,UOCC95,UIND,HIGRADED,ENUMDIST,METAREAD,MTONGUED,EDSCOR50,MIGMET5,RENT,CITY,NPBOSS50,MIGCOUNTY,MIGCITY5,ERSCOR50,BPLD,FBPLD,MBPLD,VALUEH,INCWAGE"
ar_cols_type_str = "C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,C,I,C,C,C,I,C,C,C,I,I,C,C,C,C,C,C,C,C,C,C,C,C,C,C,I,C,I,I,I,C,C,I,I,C,I,I,I,C,I,C,I,I,C,C,C,I,I,C,I,C,C,C,C,C,C,C,C,C,I,C,C,I,C,C,C,C,C,C,I,C,I,C,I,C,C,I,C,C,C,I,C"
ar_num_cols = ['NCHLT5', 'SIZEPL', 'FAMSIZE', 'HIGRADE', 'EDUC', 'AGE', 'SUPDIST', 'URBPOP', 'EDUCD']
ar_cat_cols = ['SPLIT', 'URBAN', 'SEX', 'RESPONDT', 'SLREC', 'LABFORCE', 'SCHOOL', 'FARM', 'OWNERSHP', 'SPANNAME', 'METRO', 'EMPSTAT', 'HISPAN', 'RACE', 'MARST', 'GQ', 'HISPRULE', 'MIGRATE5', 'SAMESEA5', 'SAMEPLAC', 'MIGTYPE5', 'INCNONWG', 'MARRNO', 'SEA', 'OWNERSHPD', 'EMPSTATD', 'MIGRATE5D', 'CLASSWKRD', 'VETSTATD', 'COUNTY', 'HISPAND', 'RACED', 'BPL', 'MIGSEA5', 'OCC', 'IND', 'MIGPLAC5', 'GQTYPED', 'HIGRADED', 'ENUMDIST', 'METAREAD', 'MTONGUED', 'BPLD', 'FBPLD', 'MBPLD']
ar_num_min = {
    "NCHLT5": 0,
    "SIZEPL": 1,
    "FAMSIZE": 1,
    "HIGRADE": 1,
    "EDUC": 0,
    "AGE": 0,
    "SUPDIST": 0,
    "URBPOP": 0,
    "EDUCD": 0
}
ar_num_max = {
    "NCHLT5": 9,
    "SIZEPL": 90,
    "FAMSIZE": 29,
    "HIGRADE": 99,
    "EDUC": 11,
    "AGE": 135,
    "SUPDIST": 490,
    "URBPOP": 654,
    "EDUCD": 116
}

dp_methods = ["original_data", "rmckenna", "UCLANESL", "DPSyn", "gardn999", "PrivBayes"]
dp_budgets = ["0.3", "1.0", "8.0"]
dp_number = 1
# names = ["arizona", "vermont"]
names = ["vermont"]


data_dir = "/nfsdata/tianhao/dataset/Match 3/"
for name in names:
    if name == "arizona":
        num_min = ar_num_min
        num_max = ar_num_max
        cat_cols = ar_cat_cols
        num_cols = ar_num_cols
    elif name == "vermont":
        num_min = vm_num_min
        num_max = vm_num_max
        num_cols = vm_num_cols
        cat_cols = vm_cat_cols

    dtype_dict = {col: 'int' for col in num_cols}
    dtype_dict.update({col: 'str' for col in cat_cols})
    data_path = f"{data_dir}{name.capitalize()}/original_data/{name}_nonorm.csv"
    original_data = pd.read_csv(data_path, skipinitialspace=True, dtype=dtype_dict)
    print(f"{name} original_data.shape", original_data.shape)
    print("original_data.head\n", original_data.head())

    encoder = ce.OneHotEncoder(cols=cat_cols, use_cat_names=True)
    encoder.fit(original_data[cat_cols].astype(str))

    for dp_method in dp_methods:
        for dp_budget in dp_budgets:
            if dp_method == "original_data" and dp_budget != "0.3":
                continue
            elif dp_method == "original_data" and dp_budget == "0.3":
                data = original_data.copy()
                save_path = f"{data_dir}{name.capitalize()}/{dp_method}/{name}.csv"
            else:
                data_path = f"{data_dir}{name.capitalize()}/{dp_method}/{name}_{dp_budget}_{dp_number}_nonorm.csv"
                data = pd.read_csv(data_path, skipinitialspace=True, dtype=dtype_dict)
                save_path = f"{data_dir}{name.capitalize()}/{dp_method}/{name}_{dp_budget}_{dp_number}.csv"
            print(f"{name}_{dp_method}_{dp_budget}_{dp_number} noise_data.shape: ", data.shape)
            norm(data, num_cols, cat_cols, encoder, save_path)




