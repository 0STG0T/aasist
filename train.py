import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
import json
from pathlib import Path
from data_utils_csv import ASVspoofDataset, get_torchaudio_backend
from models.AASIST_LARGE import AASIST_LARGE
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import logging
from tqdm import tqdm

def setup_logging(config):
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        filename=log_dir / f"training_{config['model_config']['architecture']}.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )

def setup_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def load_config(config_path):
    with open(config_path) as f:
        config = json.load(f)
    return config

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        self.model = AASIST_LARGE(config["model_config"]).to(self.device)

        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.device]
            )

        self.criterion = nn.CrossEntropyLoss()
        self.setup_optimizer()
        self.scaler = GradScaler() if config["train_config"]["mixed_precision"] else None

    def setup_optimizer(self):
        opt_config = self.config["optim_config"]
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(opt_config["base_lr"]),
            betas=opt_config["betas"],
            weight_decay=float(opt_config["weight_decay"]),
            amsgrad=opt_config["amsgrad"] == "True"
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config["num_epochs"],
            eta_min=float(opt_config["lr_min"])
        )

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch}")
        self.optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(progress):
            data, target = data.to(self.device), target.to(self.device)

            if self.scaler is not None:
                with autocast():
                    _, output = self.model(data)
                    loss = self.criterion(output, target)
                    loss = loss / self.config["train_config"]["gradient_accumulation_steps"]

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.config["train_config"]["gradient_accumulation_steps"] == 0:
                    if self.config["train_config"]["grad_clip"] > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["train_config"]["grad_clip"]
                        )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                _, output = self.model(data)
                loss = self.criterion(output, target)
                loss = loss / self.config["train_config"]["gradient_accumulation_steps"]
                loss.backward()

                if (batch_idx + 1) % self.config["train_config"]["gradient_accumulation_steps"] == 0:
                    if self.config["train_config"]["grad_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config["train_config"]["grad_clip"]
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            total_loss += loss.item() * self.config["train_config"]["gradient_accumulation_steps"]
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.config["train_config"]["log_steps"] == 0:
                progress.set_postfix({
                    'loss': total_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })

        return total_loss / len(train_loader), 100. * correct / total

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                _, output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)


        return val_loss / len(val_loader), 100. * correct / total

    def save_checkpoint(self, epoch, val_loss):
        if dist.get_rank() == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.module.state_dict() if isinstance(
                    self.model, DistributedDataParallel) else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss
            }
            torch.save(
                checkpoint,
                f"checkpoints/AASIST_LARGE_epoch_{epoch}.pth"
            )

def main():
    config = load_config("config/AASIST_LARGE.conf")
    setup_logging(config)
    setup_distributed()

    # Setup audio backend
    get_torchaudio_backend()

    # Create datasets
    train_dataset = ASVspoofDataset(config["train_csv_path"])
    val_dataset = ASVspoofDataset(config["val_csv_path"])

    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    trainer = Trainer(config)
    best_val_loss = float('inf')

    for epoch in range(config["num_epochs"]):
        train_sampler.set_epoch(epoch)
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        val_loss, val_acc = trainer.validate(val_loader)
        trainer.scheduler.step()

        if dist.get_rank() == 0:
            logging.info(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, "
                f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.2f}%"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                trainer.save_checkpoint(epoch, val_loss)

if __name__ == "__main__":
    main()
