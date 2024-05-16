python -m vllm.entrypoints.openai.api_server  --model casperhansen/llama-3-70b-instruct-awq --tensor-parallel-size 2 --quantization awq --enable-lora --lora-modules test_lora=llama-sft-qlora-fsdp-vllm

curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
"model": "test_lora",
"messages": [
{"role": "system", "content": "You are a helpful assistant."},
{"role": "user", "content": "Who won the world series in 2020?"}
]
}'