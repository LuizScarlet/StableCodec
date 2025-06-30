import os
import math
import glob
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms

import diffusers
from diffusers.utils.import_utils import is_xformers_available

from my_utils.testing_utils import parse_args_testing
from color_fix import adain_color_fix_quant
from StableCodec import StableCodec
from PIL import Image
from compress_utils import *


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor


def compress_one_image(net, bin_path, ori_h, ori_w, img_name, x):
    with torch.no_grad():
        output_dict = net.compress(x)
    shape = output_dict["shape"]
    if not os.path.exists(bin_path): os.makedirs(bin_path)
    output = os.path.join(bin_path, img_name)
    with Path(output).open("wb") as f:
        write_body(f, shape, output_dict["strings"])
    size = filesize(output)
    bpp = float(size) * 8 / (ori_h * ori_w)
    return bpp


def decompress_one_image(net, bin_path, ori_h, ori_w, img_name, prompt):
    output = os.path.join(bin_path, img_name)
    with Path(output).open("rb") as f:
        strings, shape = read_body(f)
    with torch.no_grad():
        out_img = net.decompress(strings, shape, prompt)
    out_img = out_img[:, :, 0 : ori_h, 0 : ori_w]
    out_img = (out_img * 0.5 + 0.5).float().cpu().detach()
    return out_img


def main(args):

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path

    if args.seed is not None:
        set_seed(args.seed)

    net = StableCodec(sd_path=sd_path, args=args)
    net.cuda().eval()
    net.codec.update(force=True)

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
      
    bpp = []
    pos_tag_prompt = [1]
    images = glob.glob(args.img_path + '/*.png')
    print(f'\nFind {str(len(images))} images in {args.img_path}\n')
    
    for img_path in images:

        print('[Processing]', img_path)
        (path, name) = os.path.split(img_path)
        fname, ext = os.path.splitext(name)
        outf = os.path.join(args.rec_path, fname+'.png')

        img = preprocess_image(img_path, transform).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]

        pad_h = (math.ceil(ori_h / 256)) * 256 - ori_h
        pad_w = (math.ceil(ori_w / 256)) * 256 - ori_w
        img_padded = F.pad(img, pad=(0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            try:
                rate = compress_one_image(net, args.bin_path, ori_h, ori_w, fname, img_padded)
                out_img = decompress_one_image(net, args.bin_path, ori_h, ori_w, fname, pos_tag_prompt)
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(str(name))
                    print("CUDA out of memory. Continuing to next image.")
                    torch.cuda.empty_cache() 
                    continue
                else:
                    raise

        output_pil = transforms.ToPILImage()(out_img[0].clamp(0.0, 1.0))

        bpp.append(rate)
        print('[BPP]', rate)

        if args.color_fix:
            img = (img * 0.5 + 0.5).float().cpu().detach()
            im_lr_resize = transforms.ToPILImage()(img[0].clamp(0.0, 1.0))
            output_pil = adain_color_fix_quant(output_pil, im_lr_resize, 16)

        output_pil.save(outf)

    print('\n[Average BPP]', np.mean(bpp))
    

if __name__ == "__main__":
    args = parse_args_testing()
    if not os.path.exists(args.rec_path): os.makedirs(args.rec_path)
    if not os.path.exists(args.bin_path): os.makedirs(args.bin_path)
    main(args)
