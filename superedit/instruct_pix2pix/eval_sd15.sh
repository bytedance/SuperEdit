export NUM_GPUS=4
export MODEL_NAME="limingcv/SuperEdit_InstructP2P_SD15_BaseInstructDiffusion"
# export MODEL_NAME="timbrooks/instruct-pix2pix"
# export MODEL_NAME="work_dirs/pretrain_weights/InstructDiffusion_diffusers"

accelerate launch --main_process_port=23333 --num_processes=$NUM_GPUS eval/eval_instructpix2pix.py --model_path $MODEL_NAME --ckpt model --azure_endpoint "your_azure_endpoint" --api_key "your_api_key" --num_inference_steps 50 --text_guidance_scale 10 --image_guidance_scale 1.5