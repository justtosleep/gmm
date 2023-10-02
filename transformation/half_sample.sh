#!/bin/bash

mkdir -p ../log

declare -A pid_datapath
declare -A datapath_method
declare -A datapath_dataset
# datapath_method["covtype/covtype_sample50k.data"]="sample50k"
# datapath_dataset["covtype/covtype_sample50k.data"]="covtype"
# datapath_method["UCLAdult/UCLAdult_famd.data"]="famd"
# datapath_dataset["UCLAdult/UCLAdult_famd.data"]="UCLAdult"
# datapath_method["UCLAdult/UCLAdult_sparse.data"]="sparse"
# datapath_dataset["UCLAdult/UCLAdult_sparse.data"]="UCLAdult"
# datapath_method["Match 2/2006/2006-data.csv]=""
# datapath_dataset["Match 2/2006/2006-data.csv"]="2006"
# datapath_method["Match 2/2017/2017-data.csv]=""
# datapath_dataset["Match 2/2017/2017-data.csv"]="2017"
# datapath_method["UCLAdult/UCLAdult_sample1.data"]="sample1"
# datapath_dataset["UCLAdult/UCLAdult_sample1.data"]="UCLAdult"
# datapath_method["UCLAdult/UCLAdult_sample2.data"]="sample2"
# datapath_dataset["UCLAdult/UCLAdult_sample2.data"]="UCLAdult"
# datapath_method["UCLAdult/UCLAdult_norm105.data"]="norm105"
# datapath_dataset["UCLAdult/UCLAdult_norm105.data"]="UCLAdult"
# datapath_method["abalone/abalone_1hot.data"]="1hot"
# datapath_dataset["abalone/abalone_1hot.data"]="abalone"
# datapath_method["covtype/covtype.data"]=""
# datapath_dataset["covtype/covtype.data"]="covtype"
# datapath_method["Match 3/Arizona/arizona.csv"]=""
# datapath_dataset["Match 3/Arizona/arizona.csv"]="arizona"
# datapath_method["Match 3/Vermont/vermont.csv"]=""
# datapath_dataset["Match 3/Vermont/vermont.csv"]="vermont"
datapath_method["Cifar/Cifar.data"]=""
datapath_dataset["Cifar/Cifar.data"]="Cifar"
datapath_method["LabelMe/LabelMe.data"]=""
datapath_dataset["LabelMe/LabelMe.data"]="LabelMe"

for datapath in "${!datapath_method[@]}"; do
    method="${datapath_method[$datapath]}"
    dataset="${datapath_dataset[$datapath]}"
    echo "Apply half sample for '$dataset' '$method'..."
    
    #check if method is empty
    if [ -z "$method" ]; then
        nohup python half_sample.py --dataset "$dataset" --data_path "$datapath" > "../log/half_sample.out" 2>&1 &
    else
        nohup python half_sample.py --dataset "$dataset" --data_path "$datapath" --method "$method" > "../log/half_sample.out" 2>&1 &
    fi
done
