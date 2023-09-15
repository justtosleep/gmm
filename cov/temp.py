import pandas as pd
import numpy as np
import os
import sys
import argparse

#1. Read data
filenames = ["Match 2/2006/2006-data.csv", "Match 2/2017/2017-data.csv", "Match 3/Arizona/arizona.csv", "Match 3/Vermont/vermont.csv"]
for filename in filenames:
    input_path="../dataset/{}".format(filename)
    data = pd.read_csv(input_path, skipinitialspace=True)
    print("filename: ", filename)
    print("data shape: ", data.shape)
    print("data head: \n", data.head())