# RANsomCheck-Data
## Train Model
### Environment
- Ubuntu 16.04.7 LTS
- NVIDIA GeForce RTX 3060（12 GB）
- CUDA Version 10.0.130

### Conda
1. `conda create -n project python=3.7`
2. `conda activate project`
3. `conda install numpy==1.21.2 scikit-learn==1.0.2 torch==1.13.1 wandb==0.16.6`

### Wandb
- You need to setup your Wandb.
- `wandb login <your_api_key>`

### Train
- `cd <your_path>/RANsomCheck-Data/model/<GRU, LSTM, Transformer>`
- `python3 ./model_wandb.py`
