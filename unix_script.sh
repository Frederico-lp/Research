#!/bin/bash

# list of tuples containing target column, flag, and name for each dataset
# dataset, target, flag, drop_columns, flag, dataset_name
echo "Running datasets"

dataset_args=(
    ("/homes/up201904580/epileptic/Epileptic_Seizure_Recognition.csv" "y" "Unnamed" "1" "epileptic_seizure")
    ("D:/Datasets/cardiovascular/cardio.csv", "cardio", "id", "2", "cardiovascular_disease")
    ("D:/Datasets/credit/creditcard.csv", "Class", "None", "0", "credit_card_fraud")
    ("D:/Datasets/diabetes/diabetes_012_health_indicators_BRFSS2015.csv", "Diabetes_012", "None", "0", "diabetes")
    ("D:/Datasets/fetal_health/fetal_health.csv", "fetal_health", "None", "0", "fetal_health")
    ("D:/Datasets/heart_disease/heart_disease.csv", 'HeartDiseaseorAttack', "None", "0", "heart_disease")
)

for args in "${dataset_args[@]}"; do
    dataset="${args[0]}"
    echo "Running dataset ${args[4]}"
    python main.py "$dataset" "${args[1]}" "${args[2]}" "${args[3]}" "${args[4]}"
done