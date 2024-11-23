# AASIST: Audio Anti-Spoofing with Integrated Spectro-Temporal Graph Attention Networks

This repository provides an implementation of the AASIST model for audio anti-spoofing detection, with additional features for CSV-based training and batch prediction capabilities.

## Features
- CSV-based dataset support for flexible data loading
- Multi-GPU training support
- Batch prediction capabilities
- Enhanced AASIST-LARGE model variant
- Easy-to-use prediction interface

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
    --config_path config/AASIST_LARGE.conf \
    --num_gpus 2
```

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
- `path`: Absolute path to the audio file
- `target`: Label ('spoof' or 'bonafide')

## Model Architecture
- AASIST: Base model implementation
- AASIST-LARGE: Enhanced model with increased capacity
  - Doubled filter sizes
  - Increased GAT dimensions
  - Larger first convolution layer
  - Additional dropout for better regularization

## Citation
```
@inproceedings{jung2021aasist,
  title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
  author={Jung, Jee-weon and Heo, Hee-Soo and Shim, Hemlata and Yu, Ha-Jin},
  booktitle={INTERSPEECH},
  year={2021}
}
```
