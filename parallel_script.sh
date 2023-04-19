#!/bin/bash

datasets=("dataset1" "dataset2" "dataset3")  # list of dataset names

# list of tuples containing target column, flag, and name for each dataset
dataset_args=(
    ("D:/Datasets/epileptic/Epileptic_Seizure_Recognition.csv" "y" "Unnamed" "1" "epileptic_seizure")
    ("dataset2_target" "dataset2_flag" "dataset2_name")
    ("dataset3_target" "dataset3_flag" "dataset3_name")
)

# construct the command to run main.py with the correct arguments
command='python main.py {1} {2} {3} {4} {5}'

# use GNU Parallel to run the command with each dataset and its corresponding arguments
parallel --jobs 3 $command ::: "${datasets[@]}" ::: "${dataset_args[@]}"
