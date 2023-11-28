python -m torch.distributed.run --nproc_per_node=4 --master_port=11235 train.py \
   --model_name_or_path "FlagAlpha/Llama2-Chinese-7b-Chat" \
   --data_path HealthCareMagic-100k-ZHTW_Translate_change_name.json \
   --fp16 True \
   --output_dir model_output/fine_tune_zh \
   --num_train_epochs 1 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 4 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 5000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --logging_steps 1 \
   --deepspeed "./configs/optm3.json" \
   --tf32 False\
   # --model_name_or_path "daryl149/llama-2-7b-chat-hf" \
   # --deepspeed "./configs/optm3.json" \
   # --deepspeed "./configs/default_offload_opt_param.json" \
   # --lr_scheduler_type "cosine" \
   # --fsdp "full_shard offload auto_wrap" \
   # --model_name_or_path "decapoda-research/llama-7b-hf" \
   # --model_name_or_path "FlagAlpha/Llama2-Chinese-7b-Chat" \
   # --data_path ./HealthCareMagic-100k-ZHTW_Translate_all.json \
   # --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \