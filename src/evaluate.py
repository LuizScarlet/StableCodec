'''
# --------------------------------------------------------------------------------
#   (Modified) Evaluate script from MS-ILLM (https://github.com/facebookresearch/NeuralCompression/blob/main/projects/illm/eval_folder_example.py)
# --------------------------------------------------------------------------------
'''

import sys
import argparse
import tqdm
import pyiqa
import torch
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToTensor
from torchmetrics.image import (
    FrechetInceptionDistance,
    KernelInceptionDistance,
    LearnedPerceptualImagePatchSimilarity,
)
from neuralcompression.metrics import update_patch_fid

def evaluate(recon_dir, gt_dir, ntest):

    device = torch.device("cuda")
    totensor = ToTensor()

    metric_dict = {}
    # metric_dict["clipiqa"] = pyiqa.create_metric('clipiqa').to(device)
    # metric_dict["musiq"] = pyiqa.create_metric('musiq').to(device)
    # metric_dict["niqe"] = pyiqa.create_metric('niqe').to(device)
    # metric_dict["maniqa"] = pyiqa.create_metric('maniqa').to(device)
    metric_paired_dict = {}
    recon_dir = Path(recon_dir) if not isinstance(recon_dir, Path) else recon_dir
    assert recon_dir.is_dir()
    
    gt_path_list = None
    if gt_dir is not None:
        gt_dir = Path(gt_dir) if not isinstance(gt_dir, Path) else gt_dir
        gt_path_list = sorted([x for x in gt_dir.glob("*.[jpJP][pnPN]*[gG]")])
        if ntest is not None: gt_path_list = gt_path_list[:ntest]
        metric_paired_dict["psnr"] = pyiqa.create_metric('psnr').to(device)
        metric_paired_dict["dists"] = pyiqa.create_metric('dists').to(device)
        metric_paired_dict["ms_ssim"] = pyiqa.create_metric('ms_ssim').to(device)
        metric_paired_dict["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device) # lpips-alexnet 
        fid_metric = FrechetInceptionDistance().to(device)
        kid_metric = KernelInceptionDistance().to(device)
        
    recon_path_list = sorted([x for x in recon_dir.glob("*.[jpJP][pnPN]*[gG]")])

    if ntest is not None: recon_path_list = recon_path_list[:ntest]
    
    print(f'Find {len(recon_path_list)} images in {recon_dir}')
    result = {}
    for i in tqdm.tqdm(range(len(recon_path_list))):
        recon_path = str(recon_path_list[i])
        gt_path = str(gt_path_list[i]) if gt_path_list is not None else None
        
        with open(recon_path, "rb") as f:
            image_recon = Image.open(f)
            image_recon = image_recon.convert("RGB")
        recon_tensor = totensor(image_recon).unsqueeze(0).to(device)

        for key, metric in metric_dict.items():
            with torch.cuda.amp.autocast():
                value = metric(recon_tensor).item()
                result[key] = result.get(key, 0) + value
        
        if gt_dir is not None:
            with open(gt_path, "rb") as f:
                image_gt = Image.open(f)
                image_gt = image_gt.convert("RGB")
            gt_tensor = totensor(image_gt).unsqueeze(0).to(device)

            update_patch_fid(gt_tensor, recon_tensor, fid_metric=fid_metric, kid_metric=kid_metric)    
            
            for key, metric in metric_paired_dict.items():
                value = metric(recon_tensor, gt_tensor).item()
                result[key] = result.get(key, 0) + value

    
    if gt_dir is not None and len(recon_path_list) > 50:
        result['fid'] = float(fid_metric.compute())
        kid_tuple = kid_metric.compute()
        result['kid_mean'], result['kid_std'] = float(kid_tuple[0]), float(kid_tuple[1])

    print_results = []
    for key, res in result.items():
        if key == 'fid':
            print(f"{key}: {res:.2f}")
            print_results.append(f"{key}: {res:.2f}")
        elif key == 'kid_mean' or key == 'kid_std':
            print(f"{key}: {res:.7f}")
            print_results.append(f"{key}: {res:.7f}")
        else:
            print(f"{key}: {res/len(recon_path_list):.5f}")
            print_results.append(f"{key}: {res/len(recon_path_list):.5f}")
    return print_results


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example evaluation script.")
    parser.add_argument("--recon_dir", type=str)
    parser.add_argument("--gt_dir", type=str)
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    print_results = evaluate(args.recon_dir, args.gt_dir, None)

if __name__ == "__main__":
    main(sys.argv[1:])