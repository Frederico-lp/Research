#!/bin/bash

datasets=("D:/Datasets/epileptic/Epileptic_Seizure_Recognition.csv" "D:\Datasets\cardiovascular\cardio.csv" "D:\Datasets\credit\creditcard.csv")  # list of dataset names

# list of tuples containing target column, flag, and name for each dataset
dataset_args=(
    ("D:/Datasets/epileptic/Epileptic_Seizure_Recognition.csv" "y" "Unnamed" "1" "epileptic_seizure")
    ("dataset2_target", "dataset2_flag", "dataset2_name")
    ("dataset2_target", "dataset2_flag", "dataset2_name")
)

for i in "${!datasets[@]}"; do
    dataset=${datasets[$i]}
    args=${dataset_args[$i]}
    python main.py "$dataset" "${args[0]}" "${args[1]}" "${args[2]}" "${args[3]}" "${args[4]}"
done