#!/bin/bash

# Define datasets and models
datasets=("physiq" "mmfit" "spar")
models=("ex" "transformer" "unet")

# Loop over seeds, datasets, and models
for seed in {3..10}; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      python main_meta_v2.py --add_side_noise --dataset "$dataset" --model "$model" --seed "$seed"
    done
  done
done