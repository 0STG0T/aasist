#!/bin/bash\nNUM_GPUS=${1:-4}\nmkdir -p checkpoints logs\npip install torch torchaudio numpy tqdm soundfile\ntorchrun --nproc_per_node=$NUM_GPUS --master_port=29500 train.py
