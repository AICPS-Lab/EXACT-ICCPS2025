# Define datasets and models
datasets=("mmfit" "physiq" "spar")
models=("transformer" "unet" "cnn" "ex" "segmenter")

# Set a maximum number of parallel jobs
max_jobs=4  # Adjust based on your system's capacity

# Loop over seeds, datasets, and models
for seed in {42..46}; do
  for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
      # Run the command in the background
      python main_meta_v2.py --add_side_noise --dataset "$dataset" --model "$model" --seed "$seed" --n_epochs 200 &

      # Limit the number of parallel jobs
      if [[ $(jobs -r -p | wc -l) -ge $max_jobs ]]; then
        wait -n  # Wait for any job to finish before starting a new one
      fi
    done
  done
done

# Wait for all background jobs to complete
wait
