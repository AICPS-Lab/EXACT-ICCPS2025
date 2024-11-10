# EXACT
 
- Datasets folder has 4 (as of 2/5/2024) datasets of spar, spar9x, phsyiq, and opportunity (potentially unsegement), segmented
- raw_datasets folder has 3 (as of 2/5/2024) datasets of spar, spar9x, and physiq, unsegmented (has repetition of 20, 20, and 10 respectively)


exp 1: python main_meta_v2.py --model ex  --dataset physiq --add_side_noise --n_epochs 200 --window_step 25 --n_shot  5 --n_query 5

exp 2: python main_meta_v2.py --model ex  --dataset physiq --add_side_noise --n_epochs 200 --window_step 5 --loocv --n_shot  1 --n_query 1