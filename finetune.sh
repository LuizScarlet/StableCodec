#!/bin/bash

accelerate launch --num_processes=2 --gpu_ids="0,1," --main_process_port 29300 src/finetune.py \
    --sd_path="<PATH_TO_SD_TURBO>/sd-turbo" \
    --elic_path="<PATH_TO_ELIC>/elic_official.pth" \
    --codec_path="<PATH_TO_STABLECODEC>/stablecodec_base.pkl" \
    --train_dataset="<PATH_TO_DATASET>/dataset.hdf5" \
    --test_dataset="<PATH_TO_DATASET>/Kodak/" \
    --output_dir="<PATH_TO_SAVE_OUTPUTS>/" \
    --max_train_steps 21000 \
    --lambda_rate 2 # [2, 3, 4, 6, 8, 12, 16, 24, 32]