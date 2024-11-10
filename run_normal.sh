#!/bin/bash

# Define datasets and models
datasets=("mmfit")
models=("ex" "transformer" "unet")

# Loop over seeds, datasets, and models
for seed in {5..10}; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      # Check if the dataset is 'mmfit' and add --rotation_chance .5 if true
      python main_meta_v2.py --add_side_noise --dataset "$dataset" --model "$model" --seed "$seed"
    done
  done
done
