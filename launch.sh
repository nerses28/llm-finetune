# pip install flash-attn --no-build-isolation
export HF_TOKEN=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

clipsai train --data_train /clips-ai-train/datasets/ultrachat/train.jsonl --data_valid /clips-ai-train/datasets/ultrachat/validation.jsonl
clipsai align --data_train /clips-ai-train/datasets/ultrachat_dpo/train.jsonl --data_valid /clips-ai-train/datasets/ultrachat_dpo/test.jsonl --peft_path test