import sys
import safetensors.torch

src, dst = sys.argv[-2:]

tensors = safetensors.torch.load_file(f"{src}/adapter_model.safetensors")

non_lora_keys = [k for k in tensors.keys() if "lora" not in k]

print("splitting non-lora keys into a separate file")
print("lora keys: ", tensors.keys())
print("non-lora keys: ", non_lora_keys)

non_lora_tensors = {k:tensors.pop(k) for k in non_lora_keys}

safetensors.torch.save_file(tensors, f"{dst}/adapter_model.safetensors")
safetensors.torch.save_file(non_lora_tensors, f"{dst}/rest.safetensors")