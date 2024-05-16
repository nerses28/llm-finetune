dir=llama-sft-qlora-fsdp # edit to the dir with lora weights and config files
cp -r $dir $dir-vllm
python vllm-lora-convert.py $dir $dir-vllm