#!/bin/bash

accelerate launch --num_processes=2 --gpu_ids="0,1," --main_process_port 29300 src/train.py \
    --sd_path="<PATH_TO_SD_TURBO>/sd-turbo" \
    --elic_path="<PATH_TO_ELIC>/elic_official.pth" \
    --train_dataset="<PATH_TO_DATASET>/dataset.hdf5" \
    --test_dataset="<PATH_TO_DATASET>/Kodak/" \
    --output_dir="<PATH_TO_SAVE_OUTPUTS>/" \
    --max_train_steps 120000 \
    --lambda_rate 0.5