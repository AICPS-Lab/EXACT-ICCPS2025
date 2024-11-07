# run three different model: ex, transformer, and unet:

python main_meta_v2.py --add_side_noise --model ex --seed 0
python main_meta_v2.py --add_side_noise --model transformer --seed 0
python main_meta_v2.py --add_side_noise --model unet --seed 0

# make this sh file executable by running:
# chmod +x run_normal.sh