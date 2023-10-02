#!/bin/bash

mkdir -p ../log
rm -f ../log/gmm_score.out

declare -A pid_datapath
declare -A datapath_method
declare -A datapath_dataset
# datapath_method["abalone/abalone_1hot.data"]="1hot"
# datapath_dataset["abalone/abalone_1hot.data"]="abalone"
# datapath_method["covtype/covtype_sample50k.data"]="sample50k"
# datapath_dataset["covtype/covtype_sample50k.data"]="covtype"
# datapath_method["covtype/covtype.data"]=""
# datapath_dataset["covtype/covtype.data"]="covtype"
# datapath_method["UCLAdult/UCLAdult_famd.data"]="famd"
# datapath_dataset["UCLAdult/UCLAdult_famd.data"]="UCLAdult"
# datapath_method["UCLAdult/UCLAdult_sparse.data"]="sparse"
# datapath_dataset["UCLAdult/UCLAdult_sparse.data"]="UCLAdult"
datapath_method["UCLAdult/UCLAdult_norm105.data"]="norm105"
datapath_dataset["UCLAdult/UCLAdult_norm105.data"]="UCLAdult"
# datapath_method["Match 2/2006/2006-data.csv]=""
# datapath_dataset["Match 2/2006/2006-data.csv"]="2006"
# datapath_method["Match 2/2017/2017-data.csv]=""
# datapath_dataset["Match 2/2017/2017-data.csv"]="2017"
# datapath_method["toydata/gaussian.data"]=""
# datapath_dataset["toydata/gaussian.data"]="gaussian"
# datapath_method["Match 3/Arizona/arizona.csv"]=""
# datapath_dataset["Match 3/Arizona/arizona.csv"]="arizona"
datapath_method["Match 3/Vermont/vermont.csv"]=""
datapath_dataset["Match 3/Vermont/vermont.csv"]="vermont"
datapath_method["Cifar/Cifar.data"]=""
datapath_dataset["Cifar/Cifar.data"]="Cifar"
datapath_method["LabelMe/LabelMe.data"]=""
datapath_dataset["LabelMe/LabelMe.data"]="LabelMe"

for datapath in "${!datapath_method[@]}"; do
    method="${datapath_method[$datapath]}"
    dataset="${datapath_dataset[$datapath]}"
    echo "Vil diff for '$dataset' '$method'..."
    
    #check if method is empty
    if [ -z "$method" ]; then
        nohup python metrics_score.py --dataset "$dataset" --data_path "$datapath" > "../log/${dataset}_metrics_score.out" 2>&1 &
    else
        nohup python metrics_score.py --dataset "$dataset" --data_path "$datapath" --method "$method" > "../log/${dataset}_metrics_score.out" 2>&1 &
    fi
    pid=$!
    pid_datapath["$pid"]="$datapath"
done

# 等待所有 GMM 任务结束
for pid in "${!pid_datapath[@]}"; do
    datapath="${pid_datapath[$pid]}"
    dataset="${datapath_dataset[$datapath]}"
    method="${datapath_method[$datapath]}"
    wait $pid
done

# # sample_nums=(200 300 400 500 600 700 800 900 1000 1200 1400 1600 1800 2000 5000 8000 10000 20000)
# sample_nums=(200)
# for sample_num in "${sample_nums[@]}"; do
#     echo "Sample $sample_num data..."
#     # python ../transformation/sample.py --sample_num $sample_num
#     for datapath in "${!datapath_method[@]}"; do
#         method="${datapath_method[$datapath]}"
#         dataset="${datapath_dataset[$datapath]}"
#         echo "Apply gmm for '$dataset' '$method'..."
        
#         #check if method is empty
#         if [ -z "$method" ]; then
#             python ../transformation/sample.py --dataset "$dataset" --data_path "$datapath" --sample_num $sample_num > "../log/sample.out" 2>&1
#             nohup python gmm_score.py --dataset "$dataset" --data_path "$datapath" > "../log/${dataset}_gmm.out" 2>&1 &
#         else
#             python ../transformation/sample.py --dataset "$dataset" --data_path "$datapath" --method "$method" --sample_num $sample_num > "../log/sample.out" 2>&1
#             nohup python gmm_score.py --dataset "$dataset" --data_path "$datapath" --method "$method" > "../log/${dataset}_gmm.out" 2>&1 &
#         fi
#         pid=$!
#         pid_datapath["$pid"]="$datapath"
#     done

#     # 等待所有 GMM 任务结束
#     for pid in "${!pid_datapath[@]}"; do
#         datapath="${pid_datapath[$pid]}"
#         dataset="${datapath_dataset[$datapath]}"
#         method="${datapath_method[$datapath]}"
#         wait $pid
#         return_code=$?
#         echo "$sample_num:$datapath:$dataset:$method:$return_code" >> ../log/gmm_score.out
#     done
# done