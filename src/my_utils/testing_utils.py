import argparse

def parse_args_testing(input_args=None):

    parser = argparse.ArgumentParser()

    # pretrained weights
    parser.add_argument("--sd_path", help="path to SD-Turbo")
    parser.add_argument("--elic_path", help="path to pretrained ELIC model")
    parser.add_argument("--codec_path", help="path to pretrained StableCodec weights", default=None)

    # testing images
    parser.add_argument("--img_path", type=str, default='/data/Kodak/')

    # output path
    parser.add_argument("--rec_path", type=str, default='/output/rec/')
    parser.add_argument("--bin_path", type=str, default='/output/bin/')

    # details about the model architecture
    parser.add_argument("--lora_rank_unet", default=32, type=int)
    parser.add_argument("--lora_rank_vae", default=16, type=int)
    parser.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    parser.add_argument("--vae_encoder_tiled_size", type=int, default=1024) 
    parser.add_argument("--latent_tiled_size", type=int, default=96) 
    parser.add_argument("--latent_tiled_overlap", type=int, default=32)
    parser.add_argument("--pos_prompt", type=str, default="A high-resolution, 8K, ultra-realistic image with sharp focus, vibrant colors, and natural lighting.")

    # testing details
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument("--color_fix", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args
