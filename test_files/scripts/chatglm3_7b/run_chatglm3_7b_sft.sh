#!/bin/bash

cur_path=`pwd`
cur_path_last_dirname=${cur_path##*/}
cd ../../
running_path=`pwd`
model_path=$running_path/weights/$cur_path_last_dirname
data_path=$running_path/data/
checkpoint_path=$running_path/checkpoint_save/$cur_path_last_dirname/sft_checkpoint
log_path=$running_path/logs/$cur_path_last_dirname/sft

# 启动训练脚本
start_time=$(date +%s)

accelerate launch src/train_bash.py \
    --stage sft \
    --model_name_or_path path_to_llama_model \
    --do_train \
    --dataset alpaca_gpt4_en \
    --template default \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir path_to_sft_checkpoint \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --fp16

wait
end_time=$(date +%s)
e2e_time=$(( $end_time - $start_time ))

# 打印端到端训练时间
echo "E2E Training Duration sec : $e2e_time"