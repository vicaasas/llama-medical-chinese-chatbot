python -m torch.distributed.run --nproc_per_node=4 --master_port=1123 train_lora.py \
  --base_model 'FlagAlpha/Llama2-Chinese-7b-Chat' \
  --data_path 'HealthCareMagic-100k-ZHTW_Translate_change_name.json' \
  --output_dir './model_output/lora_models/' \
  # --per_device_train_batch_size 4 \
  # --per_device_eval_batch_size 4 \
  # --gradient_accumulation_steps 8 \
  # --num_epochs 2 \
  # --learning_rate 3e-5 \
  # --cutoff_len 512 \
  # --val_set_size 120 \
  # --adapter_name lora
  # --batch_size 32 \
