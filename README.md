# SFT Lab

1. ```git clone https://github.com/lightshifted/sft-lab.git```
2. ```cd sft-lab```
3. ```wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh```
4. ```chmod +x Miniconda3-latest-Linux-x86_64.sh```
5. ```./Miniconda3-latest-Linux-x86_64.sh -b```
6. ```source ~/miniconda3/bin/activate```
7. ```chmod +x setup_env.sh```
8. ```./setup_env.sh```
9. ```HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ~/models/meta-llama3.1-8b-instruct \
    --local-dir-use-symlinks False```




