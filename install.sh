export HF_TOKEN=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
pip install transformers==4.41.2 datasets==2.20.0 peft==0.11.1 accelerate==0.30.1 trl==0.9.4 bitsandbytes==0.43.1
pip install flash-attn==2.5.9.post1
git config --global user.email "maximedebruyn@gmail.com"
git config --global user.name "Maxime De Bruyn"
python tmp/create_datasets.py