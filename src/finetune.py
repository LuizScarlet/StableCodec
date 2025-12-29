import os
import gc
import lpips
import numpy as np
import transformers
import torch
import torch.utils.checkpoint
import diffusers
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.datasets import ImageFolder
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

from my_utils.training_utils import parse_args_training, H5Dataset, CLIPLoss
from StableCodec import StableCodec
import vision_aided_loss

        
def main(args):

    if args.sd_path is None:
        from huggingface_hub import snapshot_download
        sd_path = snapshot_download(repo_id="stabilityai/sd-turbo")
    else:
        sd_path = args.sd_path
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    if args.seed is not None:
        set_seed(args.seed)

    train_dataset = H5Dataset(
        args.train_dataset,
        transform=transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop((args.train_patch_size, args.train_patch_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    test_dataset = ImageFolder(
        args.test_dataset,
        split="Kodak",
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.dataloader_num_workers,
        shuffle=False,
        pin_memory=True,
    )

    net = StableCodec(sd_path=sd_path, args=args)
    net.set_train()

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")
    if args.gradient_checkpointing:
        net.unet.enable_gradient_checkpointing()
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    net_disc = vision_aided_loss.Discriminator(cv_type='dinov2_reg', output_type='conv_multi_level', loss_type=args.gan_loss_type, device="cuda")
    net_disc = net_disc.cuda()
    net_disc.requires_grad_(True)
    net_disc.cv_ensemble.requires_grad_(False)
    net_disc.train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)
    alex_lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()
    alex_lpips.requires_grad_(False)
    # net_clip = CLIPLoss(clip_model_name='/your_local_dir/clip-vit-base-patch32').cuda()
    net_clip = CLIPLoss().cuda()

    layers_to_opt = list(net.codec.parameters())
    for n, _p in net.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net.unet.conv_in.parameters())
    for n, _p in net.vae.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    optimizer = torch.optim.AdamW(layers_to_opt, lr=5e-5, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon)
    optimizer_disc = torch.optim.AdamW(net_disc.parameters(), lr=2e-5, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes)
    
    net, net_disc, optimizer, optimizer_disc, train_dataloader, lr_scheduler_disc = accelerator.prepare(net, net_disc, optimizer, optimizer_disc, train_dataloader, lr_scheduler_disc)
    net_lpips, alex_lpips, net_clip = accelerator.prepare(net_lpips, alex_lpips, net_clip)

    if accelerator.is_main_process:
        main_device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    net.to(accelerator.device, dtype=weight_dtype)
    net_lpips.to(accelerator.device, dtype=weight_dtype)
    alex_lpips.to(accelerator.device, dtype=weight_dtype)
    net_clip.to(accelerator.device, dtype=weight_dtype)

    net_disc.to(accelerator.device, dtype=weight_dtype)
    for name, module in net_disc.named_modules():
        if "attn" in name:
            module.fused_attn = False

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps", disable=not accelerator.is_local_main_process,)
    mse_loss = torch.nn.MSELoss()

    global_step = 0
    while global_step < args.max_train_steps:
        for batch in train_dataloader:

            if global_step == 5000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 2e-5
            if global_step == 10000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-5
            if global_step == 15000:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-6
            l_acc = [net, net_disc]

            with accelerator.accumulate(*l_acc):
                B, C, H, W = batch.shape
                pos_tag_prompt = [1 for _ in range(B)]
                x_hat, RateLossOutput = net(batch, pos_tag_prompt, args.train_patch_size, args.train_patch_size)
                x = batch.detach().float()
                x_hat = x_hat.float()

                loss_l2 = mse_loss(x_hat, x)
                loss_lpips = net_lpips(x_hat, x).mean()
                loss_clip = net_clip(x_hat, x)
                loss_adv = net_disc(x_hat, for_G=True).mean()
                loss_D = loss_l2 * args.lambda_l2 + loss_lpips * args.lambda_lpips + loss_clip * args.lambda_clip + loss_adv * args.lambda_gan
                loss = RateLossOutput.rate_loss + loss_D

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                loss_real = net_disc(x.detach(), for_real=True).mean() * args.lambda_gan
                accelerator.backward(loss_real)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

                loss_fake = net_disc(x_hat.detach(), for_real=False).mean() * args.lambda_gan
                accelerator.backward(loss_fake)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:

                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"stablecodec_ft{int(args.lambda_rate)}_{global_step}.pkl")
                        accelerator.unwrap_model(net).save_model(outf)

                    if global_step % args.eval_freq == 1:
                        l_rate, l_y, l_z, l_psnr, l_lpips = [], [], [], [], []
                        val_count = 0
                        for id, batch_val in enumerate(test_dataloader):
                            batch_val = batch_val.to(main_device)
                            B, C, H, W = batch_val.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                pos_tag_prompt = [1 for _ in range(B)]
                                x_hat, RateLossOutput = accelerator.unwrap_model(net)(batch_val, pos_tag_prompt, H, W)

                                x = (batch_val * 0.5 + 0.5).float()
                                x_hat = (x_hat * 0.5 + 0.5).float()

                                loss_l2 = mse_loss(x_hat, x)
                                loss_psnr = 10 * (-torch.log(loss_l2) / np.log(10))
                                loss_lpips = alex_lpips(x_hat, x)

                                loss_R = RateLossOutput.quantized_total_bpp.detach()
                                loss_yrate = RateLossOutput.quantized_latent_bpp.detach()
                                loss_zrate = RateLossOutput.quantized_hyper_bpp.detach()

                                l_rate.append(loss_R.item())
                                l_y.append(loss_yrate.item())
                                l_z.append(loss_zrate.item())
                                l_lpips.append(loss_lpips.item())
                                l_psnr.append(loss_psnr.item())

                            if args.save_val and val_count < args.save_num:
                                x = x.cpu().detach()
                                x_hat = x_hat.cpu().detach()
                                combined = torch.cat([x, x_hat], dim=3)
                                output_pil = transforms.ToPILImage()(combined[0].clamp(0.0, 1.0))
                                outf = os.path.join(args.output_dir, "eval", f"val_{id}.png")
                                output_pil.save(outf)
                                val_count += 1

                        logs = {}
                        assert len(l_psnr) == 24
                        logs["val/rate"] = np.mean(l_rate)
                        logs["val/y"] = np.mean(l_y)
                        logs["val/z"] = np.mean(l_z)
                        logs["val/psnr"] = np.mean(l_psnr)
                        logs["val/lpips"] = np.mean(l_lpips)
                        progress_bar.set_postfix(**logs)
                        gc.collect()
                        torch.cuda.empty_cache()
                        accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_training()
    main(args)
