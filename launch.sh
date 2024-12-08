# pip install flash-attn --no-build-isolation
export HF_TOKEN= #ADD YOUR HF KEY

#export NCCL_DEBUG=INFO
#export TORCH_NCCL_BLOCKING_WAIT=1
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

#export NCCL_DEBUG=INFO
#export NCCL_P2P_LEVEL=NVL
#export NCCL_IB_DISABLE=1  
#export NCCL_SOCKET_IFNAME=eth0 

#clipsai train --data_train datasets/ultrachat/train.jsonl --data_valid datasets/ultrachat/validation.jsonl

#clipsai train --data_train datasets/ultrachat/validation.jsonl --data_valid datasets/ultrachat/validation.jsonl
clipsai predict --data_test datasets/ultrachat/test.jsonl --peft_path test/ --batch_size 1 --output_filename predictions2.jsonl


#clipsai align --data_train datasets/ultrachat_dpo/train.jsonl --data_valid datasets/ultrachat_dpo/test.jsonl --peft_path test --model_suffix experiment1
