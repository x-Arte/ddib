"""
Train a diffusion model on 2D synthetic datasets.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion, args_to_dict, \
    add_dict_to_argparser
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.lowdose_datasets import load_lowdose_data


def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating low-dose loader...")
    data = load_lowdose_data(batch_size=args.batch_size, image_path=args.image_path, logger=logger, image_size = args.image_size, num_25D = args.num_25D)

    logger.log("creating low-dose model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)



    logger.log("training the diffusion model...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        lr=1e-4,
        schedule_sampler="uniform",
        microbatch=-1,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        image_size = 360,
        fp16_scale_growth=1e-3
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        type=str,
        default="",
        help="Path to the low-dose image dataset."
    )
    parser.add_argument(
        "--dose_rate",
        type=str,
        default="",
        help="Dose rate for the low-dose image dataset."
    )
    parser.add_argument(
        "--num_25D",
        type=int,
        default="",
        help="number of images used in training"
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
