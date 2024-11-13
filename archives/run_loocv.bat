@echo off

REM Define datasets and models
set datasets=physiq
set models=ex transformer unet

for %%d in (%datasets%) do (
  for %%m in (%models%) do (
    python main_meta_v2.py --add_side_noise --dataset %%d --model %%m --loocv --seed 0
  )
)
