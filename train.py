"""
Training script for AASIST models with single GPU support
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
from pathlib import Path
from tqdm import tqdm
from data_utils_csv import ASVspoofDataset, get_torchaudio_backend
from models.AASIST_LARGE import AASIST_LARGE

def train_model(
    train_csv,
    val_csv,
    model_save_path,
    config_path="config/AASIST_LARGE.conf"
):
    """
    Train the AASIST model using CSV dataset on a single GPU
    Args:
        train_csv: Path to training CSV file
        val_csv: Path to validation CSV file
        model_save_path: Path to save trained model
        config_path: Path to model configuration file
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create datasets
    train_dataset = ASVspoofDataset(train_csv)
    val_dataset = ASVspoofDataset(val_csv)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
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

    # Initialize model on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AASIST_LARGE(config['model_config']).to(device)

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
        # Training phase
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]')
        for batch_idx, (data, target) in enumerate(train_pbar):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            _, output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = output.argmax(dim=1)
            correct_train += pred.eq(target).sum().item()
            total_train += target.size(0)

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct_train/total_train:.2f}%'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]')
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                _, output = model(data)
                loss = criterion(output, target)

                val_loss += loss.item()
                pred = output.argmax(dim=1)
                correct_val += pred.eq(target).sum().item()
                total_val += target.size(0)

                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct_val/total_val:.2f}%'
                })

        val_loss /= len(val_loader)
        val_accuracy = 100. * correct_val / total_val

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'\n✓ New best model saved! (val_loss: {val_loss:.4f})')

        print(f'\nEpoch {epoch+1} Summary:')
        print(f'Train Loss: {avg_train_loss:.4f} | Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.2f}%\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv', required=True)
    parser.add_argument('--val_csv', required=True)
    parser.add_argument('--model_save_path', required=True)
    parser.add_argument('--config_path', default='config/AASIST_LARGE.conf')

    args = parser.parse_args()
    train_model(**vars(args))
