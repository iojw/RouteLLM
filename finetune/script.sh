# Works on 4xA100-80GB.
torchrun --nproc_per_node=4 --master_port=20001 train_mem.py \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf  \
    --data_path preprocess/ \
    --bf16 True \
    --output_dir output \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True \
    --model_max_length 4096
