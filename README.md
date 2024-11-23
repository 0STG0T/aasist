# AASIST: Audio Anti-Spoofing with Integrated Spectro-Temporal Graph Attention Networks

This repository provides an implementation of the AASIST model for audio anti-spoofing detection, with additional features for CSV-based training and batch prediction capabilities.

## Features
- CSV-based dataset support for flexible data loading
- Single/Multi-GPU training support with automatic device selection
- Batch prediction capabilities with GPU acceleration
- Enhanced AASIST-LARGE model variant
- Easy-to-use prediction interface
- Automatic mixed precision (AMP) training
- Robust model checkpointing and state management
- Memory-efficient data loading with configurable workers

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Training
Train the model using CSV dataset:
```bash
python train.py \
    --train_csv path/to/train.csv \
    --val_csv path/to/val.csv \
    --model_save_path models/weights/AASIST_LARGE.pth \
    --config_path config/AASIST_LARGE.conf
```

The training script automatically:
- Selects the best available device (GPU/CPU)
- Enables automatic mixed precision when GPU is available
- Saves both model weights and full checkpoints
- Maintains best model based on validation loss

### Prediction
For single file prediction:
```bash
python predict.py \
    --model_path models/weights/AASIST_LARGE.pth \
    --config_path config/AASIST_LARGE.conf \
    --input_path path/to/audio.wav
```

For batch prediction:
```bash
python predict.py \
    --model_path models/weights/AASIST_LARGE.pth \
    --config_path config/AASIST_LARGE.conf \
    --input_path path/to/audio/directory \
    --batch_size 32
```

## CSV Format
The training and validation CSV files should have the following columns:
- `wav_path`: Absolute path to the audio file
- `label`: Integer label (0 for genuine, 1 for spoof)

## Model Architecture
- AASIST: Base model implementation
- AASIST-LARGE: Enhanced model with increased capacity
  - Doubled filter sizes in all convolutional layers
  - Increased GAT attention dimensions
  - Larger first convolution layer (64 -> 128 channels)
  - Additional dropout layers (0.5) for better regularization
  - Improved batch normalization placement

## Performance Optimizations
- Efficient data loading with automatic worker scaling
- GPU memory optimization with gradient scaling
- Automatic mixed precision training support
- Robust error handling and state management
- Configurable batch sizes for both training and inference

## Citation
```
@inproceedings{jung2021aasist,
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Shim, Hemlata and Yu, Ha-Jin},
  booktitle={INTERSPEECH},
  year={2021}
}
```
