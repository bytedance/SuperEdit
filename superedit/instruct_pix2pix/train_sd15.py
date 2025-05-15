# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been modified by ByteDance on 2025
#
# Original file was released under Apache License, Version 2.0, with the full license text
# available at Apache License, Version 2.0.
#
# This modified file is released under the same license.


import argparse
import logging
import math
import random
import os
import io

import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import PIL
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset, load_from_disk, concatenate_datasets, Dataset, DatasetDict
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel, EulerAncestralDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import deprecate
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

import wandb



logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "fusing/instructpix2pix-1000-samples": ("input_image", "edit_prompt", "edited_image"),
}
WANDB_TABLE_COL_NAMES = ["original_image", "edited_image", "edit_prompt"]

torch.autograd.set_detect_anomaly(True)


def log_validation(
    pipeline,
    args,
    accelerator,
    generator,
    val_dataset=None,
):
    assert val_dataset is not None

    logger.info(f"Running validation... \n Generating {len(val_dataset)} images")
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    image_logs = []

    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for idx, val_data in enumerate(val_dataset):
            print(f"Perform validation image {idx}")
            input_image = val_data[args.original_image_column]
            edited_image = val_data[args.edited_image_column]
            editing_prompt = val_data[args.edit_prompt_column]

            if isinstance(editing_prompt, list):
                editing_prompt = editing_prompt[0]

            input_image = input_image.resize((args.resolution, args.resolution))
            edited_image = edited_image.resize((args.resolution, args.resolution))

            edited_images = pipeline(
                editing_prompt,
                input_image,
                num_inference_steps=20,
                image_guidance_scale=1.5,
                guidance_scale=7.5,
                generator=generator,
                num_images_per_prompt=4
            ).images

            image_logs.append({
                "input_image": input_image,
                "edited_image": edited_image,
                "editing_prompt": editing_prompt,
                "output_images": edited_images,
            })

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                output_images = log["output_images"]
                editing_prompt = log["editing_prompt"]
                input_image = log["input_image"]
                edited_image = log["edited_image"]

                formatted_images.append(wandb.Image(input_image, caption="Input Image"))
                formatted_images.append(wandb.Image(edited_image, caption="GT Edited Image"))

                for output_image in output_images:
                    image = wandb.Image(output_image, caption=editing_prompt)
                    formatted_images.append(image)

            tracker.log({"validation": formatted_images})


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script for InstructPix2Pix.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--original_image_column",
        type=str,
        default="input_image",
        help="The column of the dataset containing the original image on which edits where made.",
    )
    parser.add_argument(
        "--edited_image_column",
        type=str,
        default="edited_image",
        help="The column of the dataset containing the edited image.",
    )
    parser.add_argument(
        "--edit_prompt_column",
        type=str,
        default="edit_prompt",
        help="The column of the dataset containing the edit instruction.",
    )
    parser.add_argument(
        "--inverse_edit_prompt_column",
        type=str,
        default=None,
        help="The column of the dataset containing the inverse edit instruction.",
    )
    parser.add_argument(
        "--neg_edit_prompt_column",
        type=str,
        default=None,
        help="The column of the dataset containing the inverse edit instruction.",
    )
    parser.add_argument(
        "--triplet_loss_margin",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--triplet_loss_weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--triplet_loss_start_step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--triplet_loss_threshold",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stop_pos_grad",
        action="store_true",
        help="Whether to stop gradient on postive distance of triplet loss"
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run fine-tuning validation every X epochs. The validation process consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="instruct-pix2pix-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--lr_unet",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--conditioning_dropout_prob",
        type=float,
        default=None,
        help="Conditioning dropout probability. Drops out the conditionings (image and edit prompt) used in training InstructPix2Pix. See section 3.2.1 in the paper: https://arxiv.org/abs/2211.09800.",
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--keep_in_memory", action="store_true", help="Whether to keep data into memory.")
    parser.add_argument(
        "--non_ema_revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained non-ema model identifier. Must be a branch, tag or git identifier of the local or"
            " remote repository specified with --pretrained_model_name_or_path."
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")

    # default to using the same revision for the non-ema model if not specified
    if args.non_ema_revision is None:
        args.non_ema_revision = args.revision

    return args


def convert_to_np(image, resolution):

    if type(image) == bytes:
        image = PIL.Image.open(io.BytesIO(image))

    image = image.convert("RGB").resize((resolution, resolution))
    return np.array(image).transpose(2, 0, 1)


def download_image(url):
    image = PIL.Image.open(requests.get(url, stream=True).raw)
    image = PIL.ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    return image

def load_local_image(path):
    image = PIL.Image.open(path).convert("RGB")
    image = PIL.ImageOps.exif_transpose(image)
    return image


def main():
    args = parse_args()
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    if args.non_ema_revision is not None:
        deprecate(
            "non_ema_revision!=None",
            "0.15.0",
            message=(
                "Downloading 'non_ema' weights from revision branches of the Hub is deprecated. Please make sure to"
                " use `--variant=non_ema` instead."
            ),
        )
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.non_ema_revision
    )

    if unet.conv_in.weight.shape[1] != 8:
        # InstructPix2Pix uses an additional image for conditioning. To accommodate that,
        # it uses 8 channels (instead of 4) in the first (conv) layer of the UNet. This UNet is
        # then fine-tuned on the custom InstructPix2Pix dataset. This modified UNet is initialized
        # from the pre-trained checkpoints. For the extra channels added to the first layer, they are
        # initialized to zero.
        logger.info("Initializing the InstructPix2Pix UNet from the pretrained UNet.")
        in_channels = 8
        out_channels = unet.conv_in.out_channels
        unet.register_to_config(in_channels=in_channels)

        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels,                # 8 for SD1.5
                out_channels,               # 320 for SD1.5
                unet.conv_in.kernel_size,   # (3, 3) for SD1.5
                unet.conv_in.stride,        # (1, 1) for SD1.5
                unet.conv_in.padding        # (1, 1) for SD1.5
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
            unet.conv_in = new_conv_in

    # Whether to freeze vae, unet and text_encoder
    vae.requires_grad_(False)

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)


    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

                for i, model in enumerate(models):
                    print(type(model))
                    if isinstance(model, UNet2DConditionModel):
                        model.save_pretrained(os.path.join(output_dir, f"unet"))

                    if weights:
                        weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DConditionModel)
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                model = models.pop()

                if isinstance(model, UNet2DConditionModel):
                    load_model = UNet2DConditionModel.from_pretrained(os.path.join(input_dir, f"unet"))
                    model.register_to_config(**load_model.config)
                    model.load_state_dict(load_model.state_dict())
                    del load_model
                elif isinstance(model, CLIPTextModel):
                    load_model = CLIPTextModel.from_pretrained(os.path.join(input_dir, f"text_encoder"))
                    model.config = load_model.config
                    model.load_state_dict(load_model.state_dict())
                    del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.lr_unet = (
            args.lr_unet * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    param_groups = []
    param_groups.append({
        'params': unet.parameters(),
        'lr': args.lr_unet
    })

    optimizer = optimizer_cls(
        param_groups,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    with accelerator.main_process_first():
        if args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            if os.path.exists(args.dataset_name):
                dataset = load_from_disk(args.dataset_name, keep_in_memory=args.keep_in_memory)
            else:
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config_name,
                    cache_dir=args.cache_dir,
                    keep_in_memory=args.keep_in_memory,
                    num_proc=32
                )

            if isinstance(dataset, DatasetDict):
                if 'val' in dataset.keys():
                    val_dataset = dataset['val']
                else:
                    dataset = dataset['train'].train_test_split(test_size=10)
                    train_dataset, val_dataset = dataset['train'], dataset['test']
            elif isinstance(dataset, Dataset):
                dataset = dataset.train_test_split(test_size=10)
                train_dataset, val_dataset = dataset['train'], dataset['test']
        else:
            data_files = {}
            if args.train_data_dir is not None:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=args.cache_dir,
            )
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/main/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.original_image_column is None:
        original_image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        original_image_column = args.original_image_column
        if original_image_column not in column_names:
            raise ValueError(
                f"--original_image_column' value '{args.original_image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.edit_prompt_column is None:
        edit_prompt_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        edit_prompt_column = args.edit_prompt_column
        if edit_prompt_column not in column_names:
            raise ValueError(
                f"--edit_prompt_column' value '{args.edit_prompt_column}' needs to be one of: {', '.join(column_names)}"
            )
    if args.inverse_edit_prompt_column is not None:
        inverse_edit_prompt_column = args.inverse_edit_prompt_column
    else:
        inverse_edit_prompt_column = None
    if args.neg_edit_prompt_column is not None:
        neg_edit_prompt_column = args.neg_edit_prompt_column
    else:
        neg_edit_prompt_column = None
    if args.edited_image_column is None:
        edited_image_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        edited_image_column = args.edited_image_column
        if edited_image_column not in column_names:
            raise ValueError(
                f"--edited_image_column' value '{args.edited_image_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(captions):
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        ]
    )

    def preprocess_images(examples):
        original_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[original_image_column]]
        )
        edited_images = np.concatenate(
            [convert_to_np(image, args.resolution) for image in examples[edited_image_column]]
        )
        # We need to ensure that the original and the edited images undergo the same
        # augmentation transforms.
        images = np.concatenate([original_images, edited_images])
        images = torch.tensor(images)
        images = 2 * (images / 255) - 1
        return train_transforms(images)

    def preprocess_train(examples):
        # Preprocess images.
        preprocessed_images = preprocess_images(examples)
        # Since the original and edited images were concatenated before
        # applying the transformations, we need to separate them and reshape
        # them accordingly.
        original_images, edited_images = preprocessed_images.chunk(2)
        original_images = original_images.reshape(-1, 3, args.resolution, args.resolution)
        edited_images = edited_images.reshape(-1, 3, args.resolution, args.resolution)

        if neg_edit_prompt_column is not None:
            # Preprocess the negative captions.
            captions_per_sample = examples[neg_edit_prompt_column]
            # Randomly pick a worng editing prompt if the input is a list
            if isinstance(captions_per_sample[0], list):
                captions_per_sample = [random.choice(sub_list) for sub_list in captions_per_sample]

            captions = list(captions_per_sample)
            examples["neg_input_ids"] = tokenize_captions(captions)

        # reverse the input image and the edited image
        if inverse_edit_prompt_column is not None and random.random() > 0.5:
            # Collate the preprocessed images into the `examples`.
            examples["original_pixel_values"] = edited_images
            examples["edited_pixel_values"] = original_images

            # Preprocess the captions.
            captions_per_sample = examples[inverse_edit_prompt_column]
            if isinstance(captions_per_sample[0], list):
                captions_per_sample = [random.choice(sub_list) for sub_list in captions_per_sample]

            captions = list(captions_per_sample)
            examples["input_ids"] = tokenize_captions(captions)
        # normal training
        else:
            # Collate the preprocessed images into the `examples`.
            examples["original_pixel_values"] = original_images
            examples["edited_pixel_values"] = edited_images

            # Preprocess the captions.
            captions_per_sample = examples[edit_prompt_column]
            if isinstance(captions_per_sample[0], list):
                captions_per_sample = [random.choice(sub_list) for sub_list in captions_per_sample]

            captions = list(captions_per_sample)
            examples["input_ids"] = tokenize_captions(captions)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        original_pixel_values = torch.stack([example["original_pixel_values"] for example in examples])
        original_pixel_values = original_pixel_values.to(memory_format=torch.contiguous_format).float()
        edited_pixel_values = torch.stack([example["edited_pixel_values"] for example in examples])
        edited_pixel_values = edited_pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])

        if args.neg_edit_prompt_column is not None:
            neg_input_ids = torch.stack([example["neg_input_ids"] for example in examples])
        else:
            neg_input_ids = None

        return {
            "original_pixel_values": original_pixel_values,
            "edited_pixel_values": edited_pixel_values,
            "input_ids": input_ids,
            "neg_input_ids": neg_input_ids,
        }

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )

    # all the modules that may used to be prepared by accelerator
    modules_dict = {
        'unet': unet,
        'optimizer': optimizer,
        'train_dataloader': train_dataloader,
        'lr_scheduler': lr_scheduler,
    }

    # determine which module will be prepared by accelerator
    modules_to_prepare = [
        key for key, flag in [
            ('unet', True),
            ('optimizer', True),
            ('train_dataloader', True),
            ('lr_scheduler', True),
        ] if flag is True
    ]

    # Prepare all the modules at once
    prepared_modules = accelerator.prepare(*[modules_dict[key] for key in modules_to_prepare])

    # unpack and load each module that needed to be trained
    for i, key in enumerate(modules_to_prepare):
        modules_dict[key] = prepared_modules[i]

    # load these modules
    unet = modules_dict['unet']
    optimizer = modules_dict['optimizer']
    train_dataloader = modules_dict['train_dataloader']
    lr_scheduler = modules_dict['lr_scheduler']

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("editing", config=vars(args), init_kwargs={"wandb": {"name": f"{args.output_dir.split('/')[-1]}"}})

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        # Reset epoch metrics at the start of each epoch
        epoch_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_triplet_loss = 0.0
        num_steps = 0

        accumulate_params = ()
        unet.train()
        accumulate_params = accumulate_params + (unet,)

        if args.neg_edit_prompt_column is not None:
            train_triplet_loss = 0.0
            train_mse_loss = 0.0
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(accumulate_params):
                # We want to learn the denoising process w.r.t the edited images which
                # are conditioned on the original image (which was edited) and the edit instruction.
                # So, first, convert images to latent space.
                latents = vae.encode(batch["edited_pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning.
                outputs_text_encoder = text_encoder(batch["input_ids"], output_hidden_states=True)
                encoder_hidden_states = outputs_text_encoder['last_hidden_state']  # (77, 768) with LayerNorm

                if args.neg_edit_prompt_column is not None:
                    neg_outputs_text_encoder = text_encoder(batch["neg_input_ids"], output_hidden_states=True)
                    neg_encoder_hidden_states = neg_outputs_text_encoder['last_hidden_state']

                # Get the additional image embedding for conditioning.
                # Instead of getting a diagonal Gaussian here, we simply take the mode.
                original_image_embeds = vae.encode(batch["original_pixel_values"].to(weight_dtype)).latent_dist.mode()

                # Conditioning dropout to support classifier-free guidance during inference. For more details
                # check out the section 3.2.1 of the original paper https://arxiv.org/abs/2211.09800.
                if args.conditioning_dropout_prob is not None:
                    random_p = torch.rand(bsz, device=latents.device, generator=generator)
                    # Sample masks for the edit prompts.
                    prompt_mask = random_p < 2 * args.conditioning_dropout_prob
                    prompt_mask = prompt_mask.reshape(bsz, 1, 1)
                    # Final text conditioning.
                    null_conditioning = text_encoder(tokenize_captions([""]).to(accelerator.device))[0]
                    encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)
                    if args.neg_edit_prompt_column is not None:
                        neg_encoder_hidden_states = torch.where(prompt_mask, null_conditioning, neg_encoder_hidden_states)

                    # Sample masks for the original images.
                    image_mask_dtype = original_image_embeds.dtype
                    image_mask = 1 - (
                        (random_p >= args.conditioning_dropout_prob).to(image_mask_dtype)
                        * (random_p < 3 * args.conditioning_dropout_prob).to(image_mask_dtype)
                    )
                    image_mask = image_mask.reshape(bsz, 1, 1, 1)
                    # Final image conditioning.
                    original_image_embeds = image_mask * original_image_embeds

                # Concatenate the `original_image_embeds` with the `noisy_latents`.
                concatenated_noisy_latents = torch.cat([noisy_latents, original_image_embeds], dim=1)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.neg_edit_prompt_column is not None:
                    combined_timesteps = torch.cat([timesteps, timesteps], dim=0)
                    combined_noisy_latents = torch.cat([concatenated_noisy_latents, concatenated_noisy_latents], dim=0)
                    combined_encoder_hidden_states = torch.cat([encoder_hidden_states, neg_encoder_hidden_states], dim=0)
                    combined_model_pred = unet(combined_noisy_latents, combined_timesteps, combined_encoder_hidden_states, return_dict=False)[0]
                    model_pred_pos, model_pred_neg = torch.chunk(combined_model_pred, 2, dim=0)

                    # positive distance calculation
                    if args.stop_pos_grad:
                        pos_distance = F.mse_loss(model_pred_pos.float().detach(), target.float(), reduction="none").mean(dim=[1,2,3])
                    else:
                        pos_distance = F.mse_loss(model_pred_pos.float(), target.float(), reduction="none").mean(dim=[1,2,3])

                    # negative distance calculation
                    neg_distance = F.mse_loss(model_pred_neg.float(), target.float(), reduction="none").mean(dim=[1,2,3])
                    # triplet loss
                    triplet_loss = torch.relu(pos_distance - neg_distance + args.triplet_loss_margin)

                    # 添加threshold筛选
                    threshold_mask = (triplet_loss >= args.triplet_loss_threshold).float()
                    triplet_loss = triplet_loss * threshold_mask

                    # calculate the triplet loss only when image condition and text condition are both available
                    triplet_loss_mask = 1 - prompt_mask.float().mean((-1, -2)) * image_mask.float().mean((-1, -2, -3))
                    eta = 1e-8
                    triplet_loss = (triplet_loss * triplet_loss_mask).sum() / (triplet_loss_mask.sum() + eta)
                    triplet_loss = torch.tensor(0.0, device=triplet_loss.device) if triplet_loss_mask.sum() == 0 else triplet_loss

                    triplet_loss_weight = args.triplet_loss_weight if step >= args.triplet_loss_start_step else 0

                    mse_loss = F.mse_loss(model_pred_pos.float(), target.float(), reduction="mean")
                    loss = mse_loss + triplet_loss_weight * triplet_loss
                else:
                    # Predict the noise residual and compute loss
                    model_pred = unet(concatenated_noisy_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Gather the losses across all processes for logging (if we use distributed training).
                if args.neg_edit_prompt_column is not None:
                    avg_triplet_loss = accelerator.gather(triplet_loss.repeat(args.train_batch_size)).mean()
                    train_triplet_loss += avg_triplet_loss.item() / args.gradient_accumulation_steps

                    avg_mse_loss = accelerator.gather(mse_loss.repeat(args.train_batch_size)).mean()
                    train_mse_loss += avg_mse_loss.item() / args.gradient_accumulation_steps

                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    params_to_clip = unet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                if args.neg_edit_prompt_column is not None:
                    accelerator.log({
                        "step/triplet_loss": train_triplet_loss,
                        "step/mse_loss": train_mse_loss,
                        "step/total_loss": train_loss
                    }, step=global_step)
                    train_triplet_loss = 0.0
                    train_mse_loss = 0.0
                    train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    # If we use deepspeed, all the rank/GPUs need to call save_ckpt func
                    if accelerator.state.distributed_type == 'DEEPSPEED' or accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            # Add these lines after loss calculation
            epoch_loss += loss.detach().item()
            if args.neg_edit_prompt_column is not None:
                epoch_mse_loss += mse_loss.detach().item()
                epoch_triplet_loss += triplet_loss.detach().item()
            num_steps += 1

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step > 0 and global_step % args.validation_steps == 0:
                    if args.use_ema:
                        # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                        ema_unet.store(unet.parameters())
                        ema_unet.copy_to(unet.parameters())

                    # The models need unwrapping because for compatibility in distributed training mode.
                    pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                        args.pretrained_model_name_or_path,
                        unet=unwrap_model(unet),
                        text_encoder=unwrap_model(text_encoder),
                        vae=unwrap_model(vae),
                        revision=args.revision,
                        variant=args.variant,
                        torch_dtype=weight_dtype,
                    )
                    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(pipeline.scheduler.config)
                    pipeline.safety_checker = None

                    log_validation(
                        pipeline,
                        args,
                        accelerator,
                        generator,
                        val_dataset,
                    )

                    if args.use_ema:
                        # Switch back to the original UNet parameters.
                        ema_unet.restore(unet.parameters())

                    del pipeline
                    torch.cuda.empty_cache()

            accelerator.wait_for_everyone()

        # Log epoch metrics at the end of each epoch
        if accelerator.is_main_process:
            avg_epoch_loss = epoch_loss / num_steps

            if args.neg_edit_prompt_column is not None:
                avg_epoch_mse_loss = epoch_mse_loss / num_steps
                avg_epoch_triplet_loss = epoch_triplet_loss / num_steps

                # Log with both epoch and global_step
                metrics = {
                    "epoch/total_loss": avg_epoch_loss,
                    "epoch/mse_loss": avg_epoch_mse_loss,
                    "epoch/triplet_loss": avg_epoch_triplet_loss,
                    "epoch": epoch,
                }
                accelerator.log(metrics, step=global_step)

                logger.info(
                    f"Epoch {epoch} (Step {global_step}): Loss = {avg_epoch_loss:.4f}, "
                    f"MSE Loss = {avg_epoch_mse_loss:.4f}, Triplet Loss = {avg_epoch_triplet_loss:.4f}"
                )
            else:
                metrics = {
                    "epoch/total_loss": avg_epoch_loss,
                    "epoch": epoch,
                }
                accelerator.log(metrics, step=global_step)

                logger.info(f"Epoch {epoch} (Step {global_step}): Loss = {avg_epoch_loss:.4f}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())

        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            text_encoder=unwrap_model(text_encoder),
            vae=unwrap_model(vae),
            unet=unwrap_model(unet),
            revision=args.revision,
            variant=args.variant,
        )
        pipeline.save_pretrained(args.output_dir)

        # accelerator.save_state(os.path.join(args.output_dir, "last"))

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
