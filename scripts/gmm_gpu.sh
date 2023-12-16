#!/bin/bash

mkdir -p ../log
rm -f ../log/gmm_score.out

declare -A pid_datapath
declare -A datapath_method
declare -A datapath_dataset
# datapath_method["abalone/abalone.data"]=""
# datapath_dataset["abalone/abalone.data"]="abalone"
# datapath_method["covtype/covtype_sample50k.data"]="sample50k"
# datapath_dataset["covtype/covtype_sample50k.data"]="covtype"
# datapath_method["covtype/covtype.data"]=""
# datapath_dataset["covtype/covtype.data"]="covtype"
# datapath_method["UCLAdult/UCLAdult_famd.data"]="famd"
# datapath_dataset["UCLAdult/UCLAdult_famd.data"]="UCLAdult"
# datapath_method["UCLAdult/UCLAdult_sparse.data"]="sparse"
# datapath_dataset["UCLAdult/UCLAdult_sparse.data"]="UCLAdult"
# datapath_method["UCLAdult/UCLAdult_norm105.data"]="norm105"
# datapath_dataset["UCLAdult/UCLAdult_norm105.data"]="UCLAdult"
# datapath_method["UCLAdult/UCLAdult_num_1hot.data"]="num_1hot"
# datapath_dataset["UCLAdult/UCLAdult_num_1hot.data"]="UCLAdult"

# datapath_method["Match 2/2006/2006-data.csv]=""
# datapath_dataset["Match 2/2006/2006-data.csv"]="2006"
# datapath_method["Match 2/2017/2017-data.csv]=""
# datapath_dataset["Match 2/2017/2017-data.csv"]="2017"
# datapath_method["toydata/gaussian.data"]=""
# datapath_dataset["toydata/gaussian.data"]="gaussian"
datapath_method["Match 3/Arizona/arizona.csv"]=""
datapath_dataset["Match 3/Arizona/arizona.csv"]="arizona"
# datapath_method["Match 3/Vermont/vermont.csv"]=""
# datapath_dataset["Match 3/Vermont/vermont.csv"]="vermont"
# datapath_method["Cifar/Cifar.data"]=""
# datapath_dataset["Cifar/Cifar.data"]="Cifar"
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
# nohup python gmm_score.py --dataset "Cifar" --data_path "Cifar/Cifar.data" > "../log/Cifar_gmm.out" 2>&1 &
for datapath in "${!datapath_method[@]}"; do
    method="${datapath_method[$datapath]}"
    dataset="${datapath_dataset[$datapath]}"
    echo "Apply gmm for '$dataset' '$method'..."
    
    #check if method is empty
    if [ -z "$method" ]; then
        nohup python gmm_gpu.py --dataset "$dataset" --data_path "$datapath" > "../log/${dataset}_gmm_gpu.out" 2>&1 &
    else
        nohup python gmm_gpu.py --dataset "$dataset" --data_path "$datapath" --method "$method" > "../log/${dataset}_gmm_gpu.out" 2>&1 &
    fi
    pid=$!
    pid_datapath["$pid"]="$datapath"
done

# # 等待所有 GMM 任务结束
# for pid in "${!pid_datapath[@]}"; do
#     datapath="${pid_datapath[$pid]}"
#     dataset="${datapath_dataset[$datapath]}"
#     method="${datapath_method[$datapath]}"
#     wait $pid
#     return_code=$?
#     echo "$sample_num:$datapath:$dataset:$method:$return_code" >> ../log/gmm_score.out
# done

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