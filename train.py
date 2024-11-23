"""
Training script for AASIST models with CSV dataset support and multi-GPU training
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import json
from pathlib import Path
from tqdm import tqdm
import psutil
import GPUtil
from data_utils_csv import ASVspoofDataset, get_torchaudio_backend
from models.AASIST_LARGE import AASIST_LARGE

def train_model(
    train_csv,
    val_csv,
    model_save_path,
    config_path="config/AASIST_LARGE.conf",
    num_gpus=2
):
    """
    Train the AASIST model using CSV dataset and distributed training
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        model_save_path: Path to save trained model
        config_path: Path to model configuration file
        num_gpus: Number of GPUs to use for training
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize distributed training
    if num_gpus > 1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(dist.get_rank())

    # Create datasets
    train_dataset = ASVspoofDataset(train_csv)
    val_dataset = ASVspoofDataset(val_csv)

    # Create data loaders with distributed sampling
    train_sampler = DistributedSampler(train_dataset) if num_gpus > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Create model
    model = AASIST_LARGE(config['model_config'])
    model = model.cuda()

    if num_gpus > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.get_rank()],
            output_device=dist.get_rank()
        )

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['optim_config']['base_lr'],
        weight_decay=config['optim_config']['weight_decay']
    )

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        if train_sampler:
            train_sampler.set_epoch(epoch)

        model.train()
        train_loss = 0

        # Create progress bar for training
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]',
                         disable=dist.get_rank() != 0)

        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()

            _, output = model(data)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Update progress bar
            if dist.get_rank() == 0:
                gpu = GPUtil.getGPUs()[dist.get_rank()]
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'GPU mem': f'{gpu.memoryUsed}MB',
                    'GPU util': f'{gpu.load*100:.1f}%'
                })

        train_pbar.close()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        correct = 0

        # Create progress bar for validation
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]',
                       disable=dist.get_rank() != 0)

        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.cuda(), target.cuda()
                _, output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

                # Update progress bar
                if dist.get_rank() == 0:
                    gpu = GPUtil.getGPUs()[dist.get_rank()]
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'GPU mem': f'{gpu.memoryUsed}MB'
                    })

        val_pbar.close()
        val_loss /= len(val_loader)
        accuracy = correct / len(val_dataset)

        # Save best model
        if val_loss < best_val_loss and dist.get_rank() == 0:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'\n✓ New best model saved! (val_loss: {val_loss:.4f})')

        if dist.get_rank() == 0:
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Accuracy: {accuracy:.4f}\n')

    if num_gpus > 1:
        dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--model_save_path', required=True)
    parser.add_argument('--config_path', default='config/AASIST_LARGE.conf')
    parser.add_argument('--num_gpus', type=int, default=2)

    args = parser.parse_args()
    train_model(**vars(args))
