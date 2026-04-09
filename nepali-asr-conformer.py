import torch
import soundfile as sf
import librosa
import argparse
from transformers import AutoModel

def load_audio(path, target_sr=16000):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio.astype("float32")

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "ai4bharat/indic-conformer-600m-multilingual"
    print("Loading AI4Bharat Conformer model...")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    print(f"Loading audio: {args.audio}")
    audio = load_audio(args.audio)
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)  # [1, T]

    print(f"Running ASR for language: {args.lang}")
    transcription = model(audio_tensor, lang=args.lang)  # Directly returns string
    print("\n===== TRANSCRIPTION =====")
    print(transcription)
    print("=========================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4Bharat Indic Conformer ASR - Nepali")
    parser.add_argument("--audio", type=str, required=True, help="Path to WAV audio file")
    parser.add_argument("--lang", type=str, default="ne", help="Language code, e.g., ne for Nepali")
    args = parser.parse_args()

    main(args)