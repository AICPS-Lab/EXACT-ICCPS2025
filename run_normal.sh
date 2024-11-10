#!/bin/bash

# Define datasets and models
datasets=("mmfit" "physiq" "spar")
models=("transformer" "unet" "cnn" "ex" "segmenter")

# Loop over seeds, datasets, and models
for seed in {43..44}; do  # Run four times with seeds 43, 44, 45, 46
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      python main_meta_v2.py --add_side_noise --dataset "$dataset" --model "$model" --seed "$seed" --n_epochs 200
    done
  done
done
