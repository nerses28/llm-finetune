import os
import json
import sys
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from pydantic import BaseModel
import torch

# Load the configuration
def load_config(config_path):
    """
    Load configuration from a JSON file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} does not exist.")
    
    with open(config_path, "r") as f:
        config = json.load(f)

    peft_path = config.get("peft_path")
    base_model_path = config.get("base_model_path")

    if not peft_path and not base_model_path:
        raise ValueError("Either 'peft_path' or 'base_model_path' must be defined in the configuration.")

    if peft_path and not os.path.exists(peft_path):
        raise FileNotFoundError(f"The specified 'peft_path' ({peft_path}) does not exist.")

    if not peft_path and base_model_path and not os.path.exists(base_model_path):
        raise FileNotFoundError(f"The specified 'base_model_path' ({base_model_path}) does not exist.")

    port = config.get("port", 8000)

    return {
        "peft_path": peft_path,
        "base_model_path": base_model_path,
        "port": port
    }

# Initialize FastAPI
app = FastAPI()

# Global variables for the model and tokenizer
model = None
tokenizer = None

@app.on_event("startup")
async def startup_event():
    """
    Load the model and tokenizer during server startup.
    """
    global model, tokenizer
    try:
        if len(sys.argv) < 2:
            raise ValueError("Usage: python server.py <config_file_path>")
        
        config_file_path = sys.argv[1]
        config = load_config(config_file_path)

        if config["peft_path"]:
            print("Loading LoRA adapter model...")
            peft_config = PeftConfig.from_pretrained(config["peft_path"])
            base_model_path = peft_config.base_model_name_or_path

            # Load the base model
            model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(model, config["peft_path"])
        else:
            print("Loading base model...")
            base_model_path = config["base_model_path"]
            model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        tokenizer.pad_token = tokenizer.eos_token
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        print("Model and tokenizer loaded successfully.")

    except Exception as e:
        print(f"Failed to load model: {e}")
        raise RuntimeError(f"Server initialization failed: {e}")

class PredictionRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50
    temperature: float = 0.7
    top_p: float = 0.9
    
@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Endpoint to generate a response based on the provided prompt.
    """
    prompt = request.prompt
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required.")

    global model, tokenizer
    if model is None or tokenizer is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0, input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return {"prompt": prompt, "response": generated_text}

@app.post("/apply_chat_template")
async def apply_chat_template_endpoint(request: dict):
    messages = request.get("messages", [])
    global tokenizer
    if not messages:
        raise HTTPException(status_code=400, detail="Messages field is required.")

    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False)
    return {"text": formatted_text}
    
# Run the server
if __name__ == "__main__":
    import uvicorn

    if len(sys.argv) < 2:
        print("Usage: python server.py <config_file_path>")
        sys.exit(1)

    config_file_path = sys.argv[1]
    config = load_config(config_file_path)

    uvicorn.run(app, host="0.0.0.0", port=config["port"])

