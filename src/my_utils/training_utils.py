import h5py
import torch
import argparse
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from transformers import CLIPVisionModelWithProjection

def parse_args_training(input_args=None):

    parser = argparse.ArgumentParser()

    # pretrained weights
    parser.add_argument("--sd_path", required=True, help="Path to SD-Turbo")
    parser.add_argument("--elic_path", required=True, help="Path to pretrained ELIC model")
    parser.add_argument("--codec_path", help="Path to pretrained StableCodec weights", default=None)

    # dataset
    parser.add_argument("--train_dataset", required=True, help="Path to training dataset (hdf5)")
    parser.add_argument("--test_dataset", required=True, help="Path to test dataset (Kodak)")

    # loss function
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.1, type=float)
    parser.add_argument("--lambda_clip", default=0.1, type=float)
    parser.add_argument("--lambda_lpips", default=1.0, type=float)
    parser.add_argument("--lambda_l2", default=2.0, type=float)
    parser.add_argument("--lambda_rate", required=True, default=0.5, type=float)

    # model details
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--train_patch_size", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--max_train_steps", type=int, default=120000)
    parser.add_argument("--checkpointing_steps", type=int, default=10000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", default=True, help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--save_val", default=True)
    parser.add_argument("--save_num", type=int, default=10, help="Number of visual samples to save")

    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


class H5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.file_path = path
        self.dataset = h5py.File(self.file_path, 'r')
        with h5py.File(self.file_path, 'r') as file:
            self.dataset_len = len(file)
        self.transform = transform
        self.keys = [key for key in self.dataset.keys() if self.dataset[key].shape[0] >= 512 and self.dataset[key].shape[1] >= 512]

    def __getitem__(self, index):
        key = self.keys[index]
        image = self.dataset[key][:]
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.keys)
    

class CLIPLoss(torch.nn.Module):

    def __init__(self, clip_model_name = "openai/clip-vit-base-patch32"):
        super().__init__()
        
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_name).eval()
        self.image_encoder.requires_grad_(False)

        self.transform_for_clip = transforms.Compose([
            transforms.Lambda(lambda x: (x + 1) / 2.0),  
            transforms.Resize(224),                     
            transforms.CenterCrop(224),                 
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),  
        ])

    def forward(self, rec, gt):

        rec_inputs = self.transform_for_clip(rec)
        gt_inputs = self.transform_for_clip(gt)

        rec_features = self.image_encoder(rec_inputs).image_embeds
        gt_features = self.image_encoder(gt_inputs).image_embeds

        rec_features = rec_features / rec_features.norm(p=2, dim=-1, keepdim=True)
        gt_features = gt_features / gt_features.norm(p=2, dim=-1, keepdim=True)

        loss = torch.norm(gt_features - rec_features, p=2, dim=-1).mean()
        return loss