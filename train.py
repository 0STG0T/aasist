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
from data_utils_csv import ASVspoofDataset
from models.AASIST_LARGE import AASIST_LARGE
from torch.amp import autocast, GradScaler

def train_model(train_csv, val_csv, model_save_path, config_path="config/AASIST_LARGE.conf"):
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Memory optimization settings
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reduce batch size and add gradient accumulation
    effective_batch_size = config['batch_size']
    actual_batch_size = effective_batch_size // 4  # Reduce memory usage
    accumulation_steps = 4

    # Create datasets
    train_dataset = ASVspoofDataset(train_csv)
    val_dataset = ASVspoofDataset(val_csv)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}\n")

    # Create data loaders with reduced batch size
    train_loader = DataLoader(
        train_dataset,
        batch_size=actual_batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=actual_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if torch.cuda.is_available() else False
    )

    # Create model
    model = AASIST_LARGE(config).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Gradient scaler for mixed precision training
    scaler = GradScaler()

    # Training loop
    best_val_loss = float('inf')
    optimizer.zero_grad()  # Initial gradient clear

    for epoch in range(config['num_epochs']):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        # Training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]")
        for batch_idx, (inputs, targets) in enumerate(train_pbar):
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accumulation_steps  # Scale loss

            scaler.scale(loss).backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            train_pbar.set_postfix({
                'loss': train_loss/(batch_idx+1),
                'acc': f'{100.*correct/total:.2f}%'
            })

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]")
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_pbar):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                val_pbar.set_postfix({
                    'loss': val_loss/(batch_idx+1),
                    'acc': f'{100.*correct/total:.2f}%'
                })

        val_loss = val_loss/len(val_loader)

        # Save best model
        if val_loss < best_val_loss:
            print(f"\n✓ New best model saved! (val_loss: {val_loss:.4f})")
            best_val_loss = val_loss

            # Save model weights
            torch.save(model.state_dict(), model_save_path)

            # Save full checkpoint
            checkpoint_dir = Path(model_save_path).parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_dir / "best_checkpoint.pth")

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Accuracy: {100.*correct/total:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {100.*correct/total:.2f}%\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True, help="Path to training CSV file")
    parser.add_argument("--val_csv", required=True, help="Path to validation CSV file")
    parser.add_argument("--model_save_path", required=True, help="Path to save model weights")
    parser.add_argument("--config_path", default="config/AASIST_LARGE.conf", help="Path to model config")
    args = parser.parse_args()

    train_model(args.train_csv, args.val_csv, args.model_save_path, args.config_path)
