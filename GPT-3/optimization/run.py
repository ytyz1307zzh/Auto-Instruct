"""
Script for running instruction selection training
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pdb
import sys
import argparse
import logging
import wandb
import random
import numpy as np
import torch
import transformers
from Logger import logger, set_log_path_and_level

from train import run

torch.set_printoptions(threshold=1000, edgeitems=6)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v in ["true", "True"]:
        return True
    elif v in ["false", "False"]:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v} instead.")


def main():
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument("--data_dir", type=str, default="./processed_data",
                        help='Path to the dataset (already processed by DatasetBuilder)')
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help='Directory to save the checkpoints')
    parser.add_argument("--overwrite_output", default=False, type=str2bool,
                        help="If `True`, will overwrite the previous checkpoints (if exist).")
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_test", action='store_true')
    parser.add_argument("--sample_train_data", default=None, type=int,
                        help='If not None, then randomly downsample training data to this number')
    parser.add_argument("--test_on_dev", default=False, type=str2bool,
                        help='If True, will perform inference on dev set')
    parser.add_argument("--test_on_train", default=False, type=str2bool,
                        help='If True, will perform inference on train set')
    parser.add_argument("--sample_test_data", default=None, type=int,
                        help='If not None, then downsample test data to this number')

    # model arguments
    parser.add_argument('--model_name', type=str, default="google/flan-t5-large",
                        help='Huggingface transformer model name')
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training (example level)")
    parser.add_argument("--train_mini_batch_size", default=None, type=int,
                        help='Mini batch size (instruction level). '
                             'If None, then the actual batch size is train_batch_size * num_instructions_per_example. '
                             'If not None, then use this value as the actual batch size, i.e., the instructions of the '
                             'same example may be encoded across multiple forward passes. This would be useful when '
                             'even train_batch_size=1 is too large for GPU memory.')
    parser.add_argument("--test_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation (example level)")
    parser.add_argument("--test_mini_batch_size", default=None, type=int,
                        help="Mini batch size (instruction level) for inference, to prevent OOM")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--constant_lr", default=True, type=str2bool,
                        help='If True, then the lr of adafactor is fixed; otherwise, it is adaptable. '
                             'Note: this argument is ignored if using adamW.')
    parser.add_argument('--optimizer', default='adafactor', type=str, choices=['adafactor', 'adamw'],
                        help='Which optimizer to use.')
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--dropout", default=None, type=float,
                        help="Dropout rate during training")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Max gradient norm.")
    parser.add_argument('--gradient_checkpoint', type=str2bool, default=False,
                        help='If true, enable gradient checkpointing to save memory (will slow down training)')
    parser.add_argument("--num_train_epochs", default=1000, type=int,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--warmup_ratio", default=0.1, type=float,
    #                     help="The number of steps used for warmup.")
    parser.add_argument('--wait_steps', type=int, default=10,
                        help="If evaluation metric does not improve for this number of global steps, "
                             "stop the training process.")
    parser.add_argument('--eval_steps', type=int, default=1000,
                        help="The evaluation interval (in global steps)")
    parser.add_argument('--rolling_loss_steps', type=int, default=100,
                        help='The interval of reporting loss (in global steps)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup ratio (only effective for adamW)')
    parser.add_argument('--output_max_length', type=int, default=2,
                        help='The max length of the generated sequence')

    # Other parameters
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', -1),
                        help='Local rank in distributed training')
    parser.add_argument('--log_level', default=logging.INFO, type=int, help="Log level on main node.")
    parser.add_argument('--log_level_replica', default=logging.WARNING, type=int, help="Log level on replica node.")
    parser.add_argument('--prefix', type=str, default='',
                        help="Prefix for saving predictions")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', default=False, type=str2bool,
                        help='If True, use fp16 mixed precision training')
    parser.add_argument('--bf16', default=False, type=str2bool,
                        help='If True, use bf16 mixed precision training')
    parser.add_argument('--wandb_project_name', type=str, default='Auto-Instruct')
    parser.add_argument('--wandb_username', type=str, default=None)

    args = parser.parse_args()

    assert not (args.test_on_dev and args.test_on_train), "These two values cannot be both True!"

    if args.fp16 and args.bf16:
        raise ValueError("Both fp16 and bf16 are set to True!")
    elif args.fp16 or args.bf16:
        args.half_precision = True
    else:
        args.half_precision = False

    # Create output_dir for logging and saving checkpoints
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Start writing logs
    log_filename = "{}log.txt".format("" if args.do_train else args.prefix)
    log_filepath = os.path.join(args.output_dir, log_filename)
    if args.local_rank == -1 or args.local_rank == 0:
        log_level = args.log_level
    else:
        log_level = args.log_level_replica
    set_log_path_and_level(log_filepath, log_level)

    transformers.utils.logging.set_verbosity(log_level)
    logger.info(args)
    logger.info(f"Save path set to {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.do_train:
        if os.path.exists(os.path.join(args.output_dir, "pytorch_model.bin")):
            logger.warning(f"Output directory {args.output_dir} already exists and has previous checkpoints!")
            if not args.overwrite_output:
                raise FileExistsError(
                    f"Output directory {args.output_dir} already exists and has previous checkpoints!")
            else:
                logger.warning("Previous checkpoints will be overwritten!")

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    if args.local_rank == -1 or args.local_rank == 0:
        with open(os.path.join(args.output_dir, "args"), 'w', encoding='utf8') as fargs:
            print(args, file=fargs)

    if args.do_train:
        if args.local_rank == -1 or args.local_rank == 0:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_username,
                name=args.output_dir.split('/')[-1],
                settings=wandb.Settings(start_method="fork")
            )
            wandb.config = args
    # During inference, do not activate wandb logging
    else:
        wandb.init(mode="disabled")

    run(args)


if __name__ == '__main__':
    main()
