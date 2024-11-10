#!/bin/bash

# Define datasets and models
datasets=("physiq" "mmfit" "spar")
# unet and ex not here:
models=( "transformer" "segmenter" "cnn")

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    if [[ "$dataset" == "spar" || "$dataset" == "mmfit" ]]; then
      window_size=300
    else
      window_size=500
    fi
    python main_meta_v2.py --add_side_noise --dataset "$dataset" --model "$model" --loocv --window_step 5 --window_size "$window_size" --n_shot 1 --n_query 1 --n_epochs 200 
  done
done