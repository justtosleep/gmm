#!/bin/bash

filenames=("adult/adult.data")
# filenames=("abalone/abalone.data" "covtype/covtype_restored.data" "adult/adult.data")
# filenames=("Match 2/2006/2006-data.csv" "Match 2/2017/2017-data.csv" "Match 3/Arizona/arizona.csv" "Match 3/Vermont/vermont.csv")
# filenames=("Match 3/Arizona/arizona.csv" "Match 3/Vermont/vermont.csv")
colstype="num" # num, cat

for filename in "${filenames[@]}"; do
    echo "Calculating covariance matrix for '$filename'..."
    path="../dataset/${filename}"
    if [ ! -f "$path" ]; then
        echo "Error: File '$path' does not exist."
        continue
    fi
    if [ "$colstype" == "mix" ]; then
        python cal_cov.py "$path"
        continue
    fi
    python cal_cov.py "$path" --colstype "$colstype"
done
