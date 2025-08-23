#!/bin/bash
export PYTHONWARNINGS="ignore::UserWarning"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING

# Wandb configuration
export WANDB_PROJECT="vlm-grpo-training"
export WANDB_RUN_NAME="grpo-qwen-7b-$(date +%Y%m%d-%H%M%S)"
export WANDB_ENTITY=""  # Your wandb username/team (optional)
export WANDB_MODE="online"  # or "offline" for local logging only
export WANDB_API_KEY="your-wandb-api-key"

deepspeed --num_gpus 8 --master_port 29517 examples/scripts/grpo_video.py \
    --deepspeed configs/ds_z3_config.json \
    --model_name_or_path /home/ec2-user/SageMaker/efs/Models/Qwen2.5-VL-7B-Instruct \
    --output_dir ./checkpoints \
    --data_dir /home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video \
    --max_samples_per_dataset 3000 \
    --do_train \
    --learning_rate 1e-6 \
    --torch_dtype bfloat16 \
    --max_prompt_length 16384 \
    --max_completion_length 512 \
    --per_device_train_batch_size 4 \
    --beta 0.1 \
    --temperature 0.8 \
    --top_p 0.9 \
    --num_generations 8 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --logging_steps 10 \
    --save_steps 50 \
    --overwrite_output_dir true \
    --bf16 true \
    --ddp_timeout 3600000 \
    --eval_strategy steps \
    --eval_steps 50 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.5 \
    --vllm_tensor_parallel_size 4 \
    --attn_implementation flash_attention_2 \
    --report_to wandb \
    --run_name "grpo-qwen-7b-$(date +%Y%m%d-%H%M%S)" \
    --log_completions \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 8 \
    --video_fps 1 \
    --video_max_frames 16 \
    # --video_min_pixels 3136 \
    # --video_max_pixels 200704 \
    # --video_total_pixels 15728640
    # --max_steps 1000 \
    # --per_device_eval_batch_size 2