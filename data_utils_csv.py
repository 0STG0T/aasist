"""
Data utilities for loading ASVspoof dataset from CSV files
"""

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

class ASVspoofDataset(Dataset):
    """Dataset class for loading audio files from CSV metadata"""

    def __init__(self, csv_path, sample_rate=16000, num_samples=64600):
        """
        Args:
            csv_path: Path to CSV file containing 'wav_path' and 'label' columns
            sample_rate: Target sample rate for audio
            num_samples: Number of samples to pad/trim to
        """
        self.df = pd.read_csv(csv_path)
        self.sample_rate = sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform, sr = torchaudio.load(row['wav_path'])

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if necessary
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        # Pad or trim to target length
        if waveform.shape[1] < self.num_samples:
            # Pad with zeros
            padding = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            # Trim to target length
            waveform = waveform[:, :self.num_samples]

        # Convert label to integer (0 for genuine, 1 for spoof)
        target = int(row['label'])

        return waveform.squeeze(0), torch.tensor(target, dtype=torch.long)

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for DataLoader"""
        waveforms, targets = zip(*batch)
        waveforms = torch.stack(waveforms)
        targets = torch.stack(targets)
        return waveforms, targets
