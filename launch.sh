# pip install flash-attn --no-build-isolation
export HF_TOKEN=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
python train.py \
--seed 100 \
--model_name_or_path "meta-llama/Meta-Llama-3-8B-Instruct" \
--train_filename "datasets/no_system_prompt/train.jsonl" \
--max_seq_len 128 \
--num_train_epochs 0.05 \
--logging_steps 5 \
--log_level "info" \
--logging_strategy "steps" \
--eval_strategy "no" \
--save_strategy "epoch" \
--hub_private_repo True \
--hub_strategy "every_save" \
--bf16 True \
--learning_rate 1e-4 \
--lr_scheduler_type "cosine" \
--weight_decay 1e-4 \
--warmup_ratio 0.1 \
--max_grad_norm 1.0 \
--output_dir "/workspace/mistral-sft-lora" \
--per_device_train_batch_size 24 \
--per_device_eval_batch_size 24 \
--gradient_accumulation_steps 1 \
--gradient_checkpointing True \
--lora_r 8 \
--lora_alpha 16 \
--lora_dropout 0.1 \
--dataloader_drop_last True