#!/bin/bash

datasets=("spar" "mmfit")
models=("segmenter" "cnn" "transformer" "unet" "ex")

for dataset in "${datasets[@]}"; do
    echo "--------- $dataset --------"
    echo " DICE | IOU | ROC-AU "
    echo "--------------------------"

    for model in "${models[@]}"; do
        if [[ "$dataset" == "spar" || "$dataset" == "mmfit" ]]; then
            window_size=300
        else
            window_size=500
        fi
        python ./test.py --dataset "$dataset" --model "$model" --window_size "$window_size"
    done

    echo "--------------------------"
    echo ""
done



