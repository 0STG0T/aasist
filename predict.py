import torch
import torchaudio
import argparse
from pathlib import Path
from models.AASIST import AASIST
from models.AASIST_LARGE import AASIST_LARGE

def load_audio(audio_path, target_length=64600):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    if waveform.shape[1] < target_length:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
    else:
        waveform = waveform[:, :target_length]
    return waveform

def load_model(model_path, model_type='base'):
    if model_type == 'large':
        model = AASIST_LARGE()
    else:
        model = AASIST()
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    return model

def predict_batch(model, wav_paths, device='cuda', batch_size=32):
    model = model.to(device)
    model.eval()
    results = []
    for i in range(0, len(wav_paths), batch_size):
        batch_paths = wav_paths[i:i + batch_size]
        batch_waves = torch.stack([load_audio(path) for path in batch_paths])
        batch_waves = batch_waves.to(device)
        with torch.no_grad():
            _, outputs = model(batch_waves)
            predictions = torch.sigmoid(outputs).cpu().numpy()
            results.extend(predictions.tolist())
    return results

def predict_file(model, wav_path, device='cuda'):
    model = model.to(device)
    model.eval()
    waveform = load_audio(wav_path)
    x_inp = waveform.unsqueeze(0).to(device)
    with torch.no_grad():
        _, output = model(x_inp)
        prediction = torch.sigmoid(output).cpu().numpy()[0]
    return prediction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--wav_path", required=True, help="Path to wav file or directory")
    parser.add_argument("--model_type", default="base", choices=["base", "large"], help="Model architecture type")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for directory processing")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    args = parser.parse_args()

    model = load_model(args.model_path, args.model_type)
    wav_path = Path(args.wav_path)

    if wav_path.is_file():
        result = predict_file(model, str(wav_path), args.device)
        print(f"File: {wav_path}")
        print(f"Prediction (probability of being bonafide): {result:.4f}")
    else:
        wav_files = list(wav_path.glob("**/*.wav"))
        results = predict_batch(model, [str(f) for f in wav_files], args.device, args.batch_size)
        for wav_file, result in zip(wav_files, results):
            print(f"File: {wav_file}")
            print(f"Prediction (probability of being bonafide): {result:.4f}")
