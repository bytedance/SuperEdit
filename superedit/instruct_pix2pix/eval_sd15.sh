export NUM_GPUS=4
export MODEL_NAME="limingcv/SuperEdit_InstructP2P_SD15_BaseInstructDiffusion"
# export MODEL_NAME="timbrooks/instruct-pix2pix"
# export MODEL_NAME="work_dirs/pretrain_weights/InstructDiffusion_diffusers"

# If you want to use your own trained model, change the MODEL_NAME to your own model path
# Please note we use the ema_model as the evaluation model after training
# So you need to provide the ema_model path:
    # 1. cd work_dirs/ip2p/sd15/ft-InstructDiffusion_unet1e-4_4w_512x512_iter6k_bs512-1x8x64_bf16_triplet-loss_weight-1.0-margin-5e-4-start-2k
    # 2. ln -s checkpoint-6000/unet_ema ./unet


accelerate launch --main_process_port=23333 --num_processes=$NUM_GPUS eval/eval_instructpix2pix.py --model_path $MODEL_NAME --ckpt model --azure_endpoint "your_azure_endpoint" --api_key "your_api_key" --num_inference_steps 50 --text_guidance_scale 10 --image_guidance_scale 1.5