#!/bin/bash

mkdir -p log

declare -A pid_dataset
declare -A dataset_method
while IFS=":" read -r datapath dataset method return_code; do
    if [ "$return_code" -eq 0 ]; then
        dataset_method["$dataset"]="$method"
        if [ -z "$method" ]; then
            nohup python metric.py --dataset "$dataset" --data_path "$datapath" > "../log/${dataset}_metric.out" 2>&1 &
        else
            nohup python metric.py --dataset "$dataset" --data_path "$datapath" --method "$method" > "../log/${dataset}_metric.out" 2>&1 &
        fi
        pid=$!
        pid_dataset["$pid"]="$dataset"
    fi
done < ../log/gmm.out

for pid in "${!pid_dataset[@]}"; do
    dataset="${pid_dataset[$pid]}"
    method="${dataset_method[$dataset]}"
    wait $pid
    return_code=$?
    echo "$dataset:$method:$return_code" >> ../log/metric.out
done
