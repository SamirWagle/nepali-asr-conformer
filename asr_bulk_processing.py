import os
import csv
import torch
import soundfile as sf
import librosa
import argparse
from pathlib import Path
from tqdm import tqdm
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

    input_dir = Path(args.input_dir)
    print(f"Scanning directory: {input_dir}")
    
    # Process both the main directory and any immediate subdirectories
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not subdirs:
        subdirs = [input_dir]

    with torch.no_grad():
        for subdir in subdirs:
            wav_files = sorted(list(subdir.glob("*.wav")))
            if not wav_files:
                continue
                
            print(f"\nProcessing folder: {subdir.name} ({len(wav_files)} files)")
            output_csv = subdir / "transcriptions.csv"
            
            # Load existing transcriptions to support resumability
            existing_files = set()
            if output_csv.exists():
                with open(output_csv, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row:
                            existing_files.add(row[0])
            
            with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                # Write header if file is entirely new
                if not existing_files:
                    writer.writerow(["name", "transcription"])
                
                for wav_path in tqdm(wav_files, desc=subdir.name):
                    file_name = wav_path.name
                    # Skip if already transcribed
                    if file_name in existing_files:
                        continue
                    
                    try:
                        audio = load_audio(wav_path)
                        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        transcription = model(audio_tensor, lang=args.lang)
                        # The model might return a list or string, ensure it's extracted cleanly
                        if isinstance(transcription, list):
                            transcription_text = transcription[0]
                        else:
                            transcription_text = transcription
                            
                        writer.writerow([file_name, transcription_text])
                        f.flush() # Force write to disk immediately (resumable)
                    except Exception as e:
                        print(f"\nError processing {file_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4Bharat Indic Conformer ASR - Batch Processing")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing subfolders of WAV files")
    parser.add_argument("--lang", type=str, default="ne", help="Language code, e.g., ne for Nepali")
    args = parser.parse_args()

    main(args)
