"""
Prediction script for AASIST models
"""

import os
import torch
import torchaudio
import json
from pathlib import Path
from models.AASIST_LARGE import AASIST_LARGE

def load_model(model_path, config_path="config/AASIST_LARGE.conf", device=None):
    """Load model from path"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(config_path) as f:
        config = json.load(f)

    model = AASIST_LARGE(config).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict_file(model, audio_path, device=None):
    """Predict single audio file"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and preprocess audio
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)

    # Pad or trim
    if waveform.shape[1] < 64600:
        padding = 64600 - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :64600]

    # Predict
    x_inp = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x_inp)
        probs = torch.softmax(outputs, dim=1)
        pred = outputs.argmax(dim=1).item()
        conf = probs[0][pred].item()

    return "spoof" if pred == 1 else "genuine", conf

def predict_batch(model, audio_dir, device=None):
    """Predict all audio files in directory"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []
    for file in Path(audio_dir).glob("*.wav"):
        pred, conf = predict_file(model, str(file), device)
        results.append({
            "file": str(file),
            "prediction": pred,
            "confidence": conf
        })
        print(f"File: {file}")
        print(f"Prediction: {pred}")
        print(f"Confidence: {conf:.4f}")
        print("-" * 50)

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model weights")
    parser.add_argument("--config_path", default="config/AASIST_LARGE.conf", help="Path to model config")
    parser.add_argument("--input_path", required=True, help="Path to audio file or directory")
    parser.add_argument("--device", default=None, help="Device to use (cpu/cuda)")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    model = load_model(args.model_path, args.config_path, device)

    if os.path.isfile(args.input_path):
        pred, conf = predict_file(model, args.input_path, device)
        print(f"File: {args.input_path}")
        print(f"Prediction: {pred}")
        print(f"Confidence: {conf:.4f}")
    else:
        predict_batch(model, args.input_path, device)
