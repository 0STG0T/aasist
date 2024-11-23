"""
Prediction script for AASIST models with batch processing support
"""

import torch
import torchaudio
import argparse
from pathlib import Path
from models.AASIST import Model as AASIST
from models.AASIST_LARGE import AASIST_LARGE
import json

def load_model(model_path, config_path, device='cuda'):
    """Load trained model"""
    with open(config_path) as f:
        config = json.load(f)

    # Select model architecture
    model_class = AASIST_LARGE if config['model_config']['architecture'] == 'AASIST_LARGE' else AASIST
    model = model_class(config['model_config']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def process_audio(file_path, num_samples=64600, sample_rate=16000):
    """Load and preprocess audio file"""
    waveform, sr = torchaudio.load(file_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(sr, sample_rate)(waveform)

    # Pad or trim
    if waveform.shape[1] < num_samples:
        padding = num_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        waveform = waveform[:, :num_samples]

    return waveform.squeeze(0)

def predict_file(model, file_path, device='cuda'):
    """Predict single audio file"""
    waveform = process_audio(file_path)
    waveform = waveform.unsqueeze(0).to(device)

    probs, preds = model.predict(waveform)
    return {
        'probabilities': probs[0].cpu().numpy(),
        'prediction': 'spoof' if preds[0].item() == 1 else 'bonafide',
        'confidence': probs[0][preds[0]].item()
    }

def predict_batch(model, file_paths, batch_size=32, device='cuda'):
    """Predict batch of audio files"""
    results = []
    current_batch = []

    for file_path in file_paths:
        waveform = process_audio(file_path)
        current_batch.append(waveform)

        if len(current_batch) == batch_size:
            batch_tensor = torch.stack(current_batch).to(device)
            probs, preds = model.predict_batch(batch_tensor)

            for i, (prob, pred) in enumerate(zip(probs, preds)):
                results.append({
                    'file': file_paths[len(results) + i],
                    'probabilities': prob.cpu().numpy(),
                    'prediction': 'spoof' if pred.item() == 1 else 'bonafide',
                    'confidence': prob[pred].item()
                })
            current_batch = []

    # Process remaining files
    if current_batch:
        batch_tensor = torch.stack(current_batch).to(device)
        probs, preds = model.predict_batch(batch_tensor)

        for i, (prob, pred) in enumerate(zip(probs, preds)):
            results.append({
                'file': file_paths[len(results) + i],
                'probabilities': prob.cpu().numpy(),
                'prediction': 'spoof' if pred.item() == 1 else 'bonafide',
                'confidence': prob[pred].item()
            })

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AASIST Prediction Script')
    parser.add_argument('--model_path', required=True, help='Path to model weights')
    parser.add_argument('--config_path', required=True, help='Path to model config')
    parser.add_argument('--input_path', required=True, help='Path to audio file or directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--device', default='cuda', help='Device to run inference on')

    args = parser.parse_args()

    # Load model
    model = load_model(args.model_path, args.config_path, args.device)
    model.eval()

    input_path = Path(args.input_path)
    if input_path.is_file():
        # Single file prediction
        result = predict_file(model, input_path, args.device)
        print(f"File: {input_path}")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
    else:
        # Batch prediction for directory
        file_paths = list(input_path.glob('**/*.wav'))
        results = predict_batch(model, file_paths, args.batch_size, args.device)

        for result in results:
            print(f"File: {result['file']}")
            print(f"Prediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("-" * 50)
