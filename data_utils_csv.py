import torch
import torchaudio
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class ASVspoofDataset(Dataset):
    def __init__(self, csv_path, nb_samples=64600):
        self.df = pd.read_csv(csv_path)
        self.nb_samples = nb_samples
        self.label_map = {"spoof": 1, "bonafide": 0}

    def _load_audio(self, path):
        waveform, sample_rate = torchaudio.load(path)

        # Convert to mono if multi-channel
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)

        # Pad or truncate to nb_samples
        if waveform.shape[1] < self.nb_samples:
            padding = self.nb_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        else:
            waveform = waveform[:, :self.nb_samples]

        return waveform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        waveform = self._load_audio(row["path"])
        label = self.label_map[row["target"].lower()]
        return waveform, torch.tensor(label, dtype=torch.long)


def get_torchaudio_backend():
    if torchaudio.get_audio_backend() in ["sox_io", "soundfile"]:
        return

    available_backends = torchaudio.list_audio_backends()
    for backend in ["sox_io", "soundfile"]:
        if backend in available_backends:
            torchaudio.set_audio_backend(backend)
            return

    raise RuntimeError("No suitable audio backend found. Install sox or soundfile.")
