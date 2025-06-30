## StableCodec: Taming One-Step Diffusion for Extreme Image Compression

[![icon](https://img.shields.io/badge/ArXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2506.21977) [![python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/) [![pytorch](https://img.shields.io/badge/PyTorch-2.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/) ![visitors](https://visitor-badge.laobi.icu/badge?page_id=LuizScarlet.StableCodec)


Tianyu Zhang, Xin Luo, Li Li, Dong Liu

University of Science and Technology of China



### :hourglass: TODO
- [x] ~~Repo release~~
- [x] ~~Update paper link~~
- [x] ~~Demo~~
- [x] ~~Pretrained models~~
- [ ] Code release



## 😍 Visual Results

[<img src="assets/0805.jpg" height="300px" width="300px"/>](https://imgsli.com/MzkzNjA5)[<img src="assets/0803.jpg" height="300px" width="310px"/>](https://imgsli.com/MzkzNjEy)[<img src="assets/0880.jpg" height="300px"/>](https://imgsli.com/MzkzNjM1)

[<img src="assets/clic1.jpg" height="300px"/>](https://imgsli.com/MzkzNjU5)[<img src="assets/clic2.jpg" height="300px"/>](https://imgsli.com/MzkzNjY5)[<img src="assets/clic3.jpg" height="300px" width="184px"/>](https://imgsli.com/MzkzNjc5)

[<img src="assets/td1.jpg" width="815"/>](https://imgsli.com/MzkzNzA1)



## ⚙ Installation

```
conda create -n stablecodec python=3.10
conda activate stablecodec
pip install -r requirements.txt
```



## ⚡ Inference

**Step1: Prepare your datasets for inference**

**Step2: Download the pretrained models**

1. Download [SD-Turbo](https://huggingface.co/stabilityai/sd-turbo).
2. Download [checkpoints](https://drive.google.com/drive/folders/1itiVVAPSTATGPcHLp_bLI9r9Qi3YcM12?usp=sharing) for StableCodec and Auxiliary Encoder ([ELIC](https://arxiv.org/abs/2203.10886)):

```bash
--- List ---
stablecodec_ft2.pkl			# ~ 0.035bpp on Kodak
stablecodec_ft4.pkl			# ~ 0.025bpp on Kodak
stablecodec_ft8.pkl			# ~ 0.017bpp on Kodak
stablecodec_ft16.pkl		# ~ 0.010bpp on Kodak
stablecodec_ft32.pkl		# ~ 0.005bpp on Kodak
elic_official.pth			# Pretrained ELIC model for Auxiliary Encoder
```

**Step3: Inference for StableCodec**

Please modify the paths in `compress.sh`:

```bash
python src/compress.py \
    --sd_path="<PATH_TO_SD_TURBO>/sd-turbo" \
    --elic_path="<PATH_TO_ELIC>/elic_official.pth" \
    --img_path="<PATH_TO_DATASET>/" \
    --rec_path="<PATH_TO_SAVE_OUTPUTs>/rec/" \
    --bin_path="<PATH_TO_SAVE_OUTPUTs>/bin/" \
    --codec_path="<PATH_TO_STABLECODEC>/stablecodec_ft2.pkl" \
    # --color_fix
```

*Note: Color fix is recommended when inferring high-resolution images with tiling.* 

Then run:

```bash
bash compress.sh
```

You may find your bitstreams in specified `bin_path` and reconstructions in `rec_path` 🤗.



## 🍭 Evaluation (Optional)

Run the evaluation script to compute reconstruction metrics with `src/evaluate.py`

```bash
bash eval_folders.sh
```

Please make sure `recon_dir` and `gt_dir` are specified.



## :notebook: License

This work is licensed under MIT license.



## 🥰 Acknowledgement

This work is implemented based on [CompressAI](https://github.com/InterDigitalInc/CompressAI), [ELIC-Unofficial](https://github.com/JiangWeibeta/ELIC), [StableSR](https://github.com/IceClear/StableSR), [StableDiffusion](https://github.com/Stability-AI/stablediffusion) and [DCVC](https://github.com/microsoft/DCVC). Thanks for their awesome work!



## :envelope: Contact

If you have any questions, please feel free to drop me an email: 

- zhangtianyu@mail.ustc.edu.cn
