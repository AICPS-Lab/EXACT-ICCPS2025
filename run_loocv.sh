#!/bin/bash

# Define datasets and models
datasets=("physiq" "mmfit" "spar")
models=("ex" "transformer" "unet")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    python main_meta_v2.py --add_side_noise --dataset "$dataset" --model "$model" --loocv
  done
done