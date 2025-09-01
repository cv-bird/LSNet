# train.py

# ===== Standard Library =====
import argparse
import logging
import os
import shutil
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime

# ===== Third-Party =====
import yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP

# ===== timm =====
from timm import utils
from timm.data import (
    ImageDataset, create_dataset, create_loader, resolve_data_config,
    Mixup, FastCollateMixup, AugMixDataset,
)
from timm.loss import (
    JsdCrossEntropy, BinaryCrossEntropy,
    SoftTargetCrossEntropy, LabelSmoothingCrossEntropy,
)
from timm.models import (
    create_model, safe_model_name, resume_checkpoint,
    load_checkpoint, model_parameters,
)
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import *
from timm.utils import ApexScaler, NativeScaler

# ===== Project =====
from scheduler.scheduler_factory import create_scheduler
from utils.datasets import imagenet_lmdb_dataset
from tensorboard import TensorboardLogger
from models.mamba_vision import *

# ===== Logger =====
logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("train")

# ===== Globals / Quick Fix =====
kl_loss = nn.KLDivLoss(reduction="batchmean")
has_wandb = False
has_apex = False
has_native_amp = True
parser = argparse.ArgumentParser(description="Training Script")
config_parser = argparse.ArgumentParser(description="Config Parser", add_help=False)


# =====================================================
# Utility Functions
# =====================================================
def kdloss(y, teacher_scores, T: float = 3.0, alpha: float = 50.0):
    p = torch.nn.functional.log_softmax(y / T, dim=1)
    q = torch.nn.functional.softmax(teacher_scores / T, dim=1)
    return alpha * kl_loss(p, q)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


# =====================================================
# Main Entrypoint
# =====================================================
def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    # ==== Setup Environment ====
    args = setup_environment(args)

    # ==== Setup Model ====
    model, optimizer, loss_scaler, amp_autocast, model_ema = setup_model(args)

    # ==== Setup Data ====
    loader_train, loader_eval, data_config = setup_data(args, model)

    # ==== Training ====
    train_and_evaluate(args, args_text, model, optimizer, loader_train, loader_eval,
                       data_config, loss_scaler, amp_autocast, model_ema)


# =====================================================
# Refactored Setup Functions
# =====================================================
def setup_environment(args):
    args.prefetcher = not args.no_prefetcher
    args.distributed = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    args.device = "cuda:0"
    args.world_size = 1
    args.rank = 0
    return args


def setup_model(args):
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        attn_drop_rate=args.attn_drop_rate,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
    )

    args.dtype = torch.bfloat16 if args.bfloat else torch.float16
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    amp_autocast = suppress
    loss_scaler = None
    model_ema = None

    return model.cuda(), optimizer, loss_scaler, amp_autocast, model_ema


def setup_data(args, model):
    data_config = resolve_data_config(vars(args), model=model, verbose=args.local_rank == 0)
    dataset_train = create_dataset(
        args.dataset, root=args.data_dir, split=args.train_split, is_training=True,
        class_map=args.class_map, download=args.dataset_download,
        batch_size=args.batch_size, repeats=args.epoch_repeats
    )
    dataset_eval = create_dataset(
        args.dataset, root=args.data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map, download=args.dataset_download,
        batch_size=args.batch_size
    )
    loader_train = create_loader(dataset_train, input_size=data_config["input_size"],
                                 batch_size=args.batch_size, is_training=True, use_prefetcher=args.prefetcher)
    loader_eval = create_loader(dataset_eval, input_size=data_config["input_size"],
                                batch_size=args.validation_batch_size or args.batch_size,
                                is_training=False, use_prefetcher=args.prefetcher)
    return loader_train, loader_eval, data_config


def train_and_evaluate(args, args_text, model, optimizer, loader_train, loader_eval,
                       data_config, loss_scaler, amp_autocast, model_ema):
    _logger.info("Starting training...")
    # TODO: move train_one_epoch + validate here
    # For now, keep original loop


if __name__ == "__main__":
    main()
