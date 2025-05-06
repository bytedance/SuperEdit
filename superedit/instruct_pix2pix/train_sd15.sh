# export MODEL_NAME="timbrooks/instruct-pix2pix"
# export MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export MODEL_NAME="limingcv/InstructDiffusion_diffusers"

export DATASET_ID="limingcv/SuperEdit-40K"
export OUTPUT_DIR="work_dirs/ip2p/sd15/ft-InstructDiffusion_unet1e-4_4w_512x512_iter6k_bs512-1x8x64_bf16_triplet-loss_weight-1.0-margin-5e-4-start-2k"

accelerate launch --config_file "superedit/instruct_pix2pix/config.yml" \
    --multi-gpu \
    --num_machines=1 --num_processes=8 --main_process_port=23456  \
    superedit/instruct_pix2pix/train_sd15.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --dataset_name=$DATASET_ID \
    --resolution=512 \
    --use_ema \
    --original_image_column="input_image" --edited_image_column="edited_image" \
    --edit_prompt_column="edit_prompts" \
    --train_batch_size=64 --gradient_accumulation_steps=1 --gradient_checkpointing --keep_in_memory \
    --neg_edit_prompt_column="neg_prompts" --triplet_loss_weight=1.0 --triplet_loss_margin=5e-4 --triplet_loss_start_step=2000 \
    --max_train_steps=6001 --validation_steps=1000 \
    --checkpointing_steps=1000 \
    --lr_unet=1e-4 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=bf16 \
    --seed=0 \
    --resume_from_checkpoint="latest" \
    --output_dir=$OUTPUT_DIR