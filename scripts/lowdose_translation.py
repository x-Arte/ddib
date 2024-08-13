"""
translation from low dose domain to low dose images
"""
import argparse
import os
import pathlib

import numpy as np
import torch.distributed as dist

from common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import model_and_diffusion_defaults_2d, add_dict_to_argparser
from guided_diffusion.lowdose_datasets import load_lowdose_data, gener_image_2D
# from guided_diffusion.synthetic_datasets import scatter, heatmap, load_2d_data, Synthetic2DType
def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")

    dist_util.setup_dist()
    logger.configure()
    logger.log("starting to sample low dose data.")

    code_folder = os.getcwd()
    image_folder = os.path.join(code_folder, f"experiments/images")
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    i = args.source
    j = args.target
    total_image = args.total_image
    logger.log(f"reading models for low dose data...")

    source_dir = os.path.join(code_folder, f"models/lowdose/{i}")
    source_model, diffusion = read_model_and_diffusion(args, source_dir)

    target_dir = os.path.join(code_folder, f"models/lowdose/{j}")
    target_model, _ = read_model_and_diffusion(args, target_dir)

    image_subfolder = os.path.join(image_folder, f"translation_{i}_{j}")
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    # sources = []
    # latents = []
    # targets = []
    data = load_lowdose_data(batch_size=args.batch_size, image_path=args.image_path, logger=logger)

    for k, (source, extra) in enumerate(data):
        logger.log(f"translating {i}->{j}, image {k}, shape {source.shape}...")
        logger.log(f"device: {dist_util.dev()}")

        source = source.to(dist_util.dev())
        
        # source = np.concatenate(source, axis=0)
        source_path = os.path.join(image_subfolder, f'test_source.npy')
        np.save(source_path, source.cpu().numpy())
        source_image_path = os.path.join(image_subfolder, f'test_source.png')
        gener_image_2D(source.cpu().numpy(), source_image_path, (360,360))

        noise = diffusion.ddim_reverse_sample_loop(
            source_model, source,
            clip_denoised=False,
            device=dist_util.dev(),
        )
        logger.log(f"obtained latent representation for {source.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

        target = diffusion.ddim_sample_loop(
            target_model, (args.batch_size, 2),
            noise=noise,
            clip_denoised=False,
            device=dist_util.dev(),
        )
        logger.log(f"finished translation {target.shape}")
        # sources.append(source.cpu().numpy())
        # latents.append(noise.cpu().numpy())
        # targets.append(target.cpu().numpy())

        # source = np.concatenate(source, axis=0)
        logger.log("target shape:", target.shape)

        source = source.cpu().numpy()
        source_path = os.path.join(image_subfolder, f'source{k}.npy')
        np.save(source_path, source)
        source_image_path = os.path.join(image_subfolder, f'source{k}.png')
        gener_image_2D(source, source_image_path, (360,360), logger=logger)

        # latent = np.concatenate(latent, axis=0)
        latent = noise.cpu().numpy()
        latent_path = os.path.join(image_subfolder, f'latent{k}.npy')
        np.save(latent_path, latent)
        latent_image_path = os.path.join(image_subfolder, f'latent{k}.png')
        gener_image_2D(latent, latent_image_path, (360,360), logger=logger)

        # target = np.concatenate(target, axis=0)
        target = target.cpu().numpy()
        target_path = os.path.join(image_subfolder, f'target{k}.npy')
        np.save(target_path, target)
        target_image_path = os.path.join(image_subfolder, f'target{k}.png')
        gener_image_2D(target, target_image_path, (360,360), logger=logger)

        dist.barrier()
        logger.log(f"image {k} translation complete")

        if k >= total_image - 1:
            break

    logger.log(f"synthetic data translation complete: {i}->{j}\n\n, total image: {total_image}")


def create_argparser():
    defaults = dict(
        num_samples=90000,
        batch_size=30000,
        model_path=""
    )
    defaults.update(model_and_diffusion_defaults_2d())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="Path to the low-dose image dataset."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Dose rate for the source low-dose image dataset."
    )
    parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Dose rate for the target low-dose image dataset."
    )
    parser.add_argument(
        "--total_image",
        type=int,
        default=6,
        help="Total number of images to generate."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()