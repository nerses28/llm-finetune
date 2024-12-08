export HF_TOKEN= #ADD YOUR HF KEY
pip3 install wandb==0.17.2 transformers==4.43.1 datasets==2.20.0 peft==0.11.1 accelerate==0.30.1 trl==0.9.4 bitsandbytes==0.43.1 fastapi==0.100.0 uvicorn==0.22.0

#pip install flash-attn==2.5.9.post1
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.0.post1/flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
pip3 install --upgrade flash_attn-2.7.0.post1+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl

git config --global user.email "maximedebruyn@gmail.com"
git config --global user.name "Maxime De Bruyn"
python3 tmp/create_datasets.py
pip3 install -e .
