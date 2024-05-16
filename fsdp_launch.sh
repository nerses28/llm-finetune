export HF_TOKEN=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
accelerate launch --config_file "fsdp_config_qlora.yaml"  train.py \
--seed 100 \
--model_name_or_path "meta-llama/Meta-Llama-3-70B-Instruct" \
--train_filename "datasets/no_system_prompt/train.jsonl" \
--chat_template_format "chatml" \
--add_special_tokens False \
--append_concat_token False \
--splits "train,test" \
--max_seq_len 128 \
--num_train_epochs 0.01 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--evaluation_strategy "no" \
--save_strategy "epoch" \
--bf16 True \
--packing True \
--learning_rate 1e-6 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.0 \
--max_grad_norm 1.0 \
--output_dir "llama-sft-qlora-fsdp" \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 2 \
--gradient_accumulation_steps 2 \
--gradient_checkpointing True \
--use_reentrant True \
--use_flash_attn False \
--use_peft_lora True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--lora_target_modules "all-linear" \
--use_4bit_quantization True \
--use_nested_quant True \
--bnb_4bit_compute_dtype "bfloat16" \
--bnb_4bit_quant_storage_dtype "bfloat16"