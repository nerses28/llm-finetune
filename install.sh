export HF_TOKEN=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
pip install transformers datasets peft accelerate==0.30.1 trl bitsandbytes
pip install flash-attn
git config --global user.email "maximedebruyn@gmail.com"
git config --global user.name "Maxime De Bruyn"
python tmp/create_datasets.py