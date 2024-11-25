# RANsomCheck-Data
## Data Processing
### Extract API Call Sequence from .json
- Extract the API call sequence from the execution report generated by Cuckoo Sandbox.
- `cd <your_path>/RANsomCheck-Data/code`
- `python3 ./extract_API_Sequence_from_json.py`

### API Call Sequence Encoding
- Convert the API call sequence into a format that can be input into the model.
- `cd <your_path>/RANsomCheck-Data/code`
- `python3 ./produce.py`

### Integrate & Split Dataset
- Combine all the encoded API call sequence datasets and split them into a training set, validation set, and test set.
- `cd <your_path>/RANsomCheck-Data/code`
- `python3 ./generate_dataset.py`

### Check Dataset
- Check the contents of the dataset and the number of samples within it.
- `cd <your_path>/RANsomCheck-Data/code`
- `python3 ./dataset_check.py`

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
