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
# datapath_method["abalone/abalone.data"]=""
# datapath_dataset["abalone/abalone.data"]="abalone"
# datapath_method["covtype/covtype.data"]=""
# datapath_dataset["covtype/covtype.data"]="covtype"
# datapath_method["Match 3/Arizona/arizona.csv"]=""
# datapath_dataset["Match 3/Arizona/arizona.csv"]="arizona"
# datapath_method["Match 3/Vermont/vermont.csv"]=""
# datapath_dataset["Match 3/Vermont/vermont.csv"]="vermont"
datapath_method["Cifar/Cifar.data"]=""
datapath_dataset["Cifar/Cifar.data"]="Cifar"
# datapath_method["LabelMe/LabelMe.data"]=""
# datapath_dataset["LabelMe/LabelMe.data"]="LabelMe"
# datapath_method["Sun/Sun.data"]=""
# datapath_dataset["Sun/Sun.data"]="Sun"
# datapath_method["Sift/Sift.data"]=""
# datapath_dataset["Sift/Sift.data"]="Sift"
# datapath_method["UKBench/UKBench.data"]=""
# datapath_dataset["UKBench/UKBench.data"]="UKBench"
# datapath_method["NUSW/NUSW.data"]=""
# datapath_dataset["NUSW/NUSW.data"]="NUSW"
# datapath_method["Tiny/Tiny.data"]=""
# datapath_dataset["Tiny/Tiny.data"]="Tiny"
# datapath_method["Trevi/Trevi.data"]=""
# datapath_dataset["Trevi/Trevi.data"]="Trevi"
# datapath_method["Gist/Gist.data"]=""
# datapath_dataset["Gist/Gist.data"]="Gist"



for datapath in "${!datapath_method[@]}"; do
    method="${datapath_method[$datapath]}"
    dataset="${datapath_dataset[$datapath]}"
    echo "Add noise to '$dataset' '$method'..."
    
    #check if method is empty
    if [ -z "$method" ]; then
        nohup python clustershift_noise.py --dataset "$dataset" --data_path "$datapath" > "../log/${dataset}_noise.out" 2>&1 &
    else
        nohup python clustershift_noise.py --dataset "$dataset" --data_path "$datapath" --method "$method" > "../log/${dataset}_noise.out" 2>&1 &
    fi
done
