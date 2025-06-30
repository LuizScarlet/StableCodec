#!/bin/bash

python src/compress.py \
    --sd_path="<PATH_TO_SD_TURBO>/sd-turbo" \
    --elic_path="<PATH_TO_ELIC>/elic_official.pth" \
    --img_path="<PATH_TO_DATASET>/" \
    --rec_path="<PATH_TO_SAVE_OUTPUTS>/rec/" \
    --bin_path="<PATH_TO_SAVE_OUTPUTS>/bin/" \
    --codec_path="<PATH_TO_STABLECODEC>/stablecodec_ft2.pkl" \
    # --color_fix