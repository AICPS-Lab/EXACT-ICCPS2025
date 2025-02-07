#!/bin/bash

datasets=("spar" "mmfit")
models=("segmenter" "cnn" "transformer" "unet"  "ex" )

for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        python ./test.py --dataset "$dataset" --model "$model" --device cpu
    done
done
