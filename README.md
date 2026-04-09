# Nepali ASR Conformer

> A simple Nepali speech-to-text CLI powered by AI4Bharat's multilingual Indic Conformer model.

[![Project Status](https://img.shields.io/badge/status-inference%20script-success)](#project-status)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#setup)
[![Model](https://img.shields.io/badge/model-AI4Bharat%20Indic%20Conformer-orange)](#model)
[![Task](https://img.shields.io/badge/task-speech%20to%20text-success)](#overview)

## Overview

`nepali-asr-conformer` is a lightweight Python script for transcribing Nepali audio using the pretrained [`ai4bharat/indic-conformer-600m-multilingual`](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) model from Hugging Face.

The current implementation focuses on inference:

- Loads an audio file from disk
- Converts stereo audio to mono
- Resamples audio to 16 kHz
- Automatically uses CUDA when available
- Loads the AI4Bharat Indic Conformer model
- Runs ASR with Nepali as the default language
- Prints the transcription in the terminal

## Project Status

This repository currently contains a single-file inference script.

```text
.
|-- .gitattributes
|-- nepali-asr-conformer.py
`-- README.md
```

Implemented:

- Audio loading with `soundfile`
- Stereo-to-mono conversion
- 16 kHz resampling with `librosa`
- Model loading through `transformers.AutoModel`
- GPU/CPU device selection with PyTorch
- CLI arguments for audio path and language code

Not implemented yet:

- Training or fine-tuning
- Batch transcription
- Dataset preparation
- WER/CER evaluation
- Transcript export to file
- Tests
- Dependency lock file

## Model

The script uses:

```python
ai4bharat/indic-conformer-600m-multilingual
```

This is loaded with:

```python
AutoModel.from_pretrained(model_name, trust_remote_code=True)
```

Because `trust_remote_code=True` is enabled, Hugging Face may execute model-specific Python code from the model repository. Use this only with model sources you trust.

## Setup

Clone the repository:

```bash
git clone https://github.com/SamirWagle/nepali-asr-conformer.git
cd nepali-asr-conformer
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Activate it on macOS or Linux:

```bash
source .venv/bin/activate
```

Install the required packages:

```bash
pip install torch transformers librosa soundfile
```

The first run may take time because the pretrained model needs to be downloaded from Hugging Face.

## Usage

Run Nepali ASR on a WAV file:

```bash
python nepali-asr-conformer.py --audio path/to/audio.wav
```

Nepali is the default language:

```bash
python nepali-asr-conformer.py --audio path/to/audio.wav --lang ne
```

Example output:

```text
Using device: cuda
Loading AI4Bharat Conformer model...
Loading audio: samples/nepali.wav
Running ASR for language: ne

===== TRANSCRIPTION =====
नमस्ते तपाईंलाई कस्तो छ
=========================
```

## CLI Arguments

| Argument | Default | Required | Description |
| --- | --- | --- | --- |
| `--audio` | none | yes | Path to the input WAV audio file |
| `--lang` | `ne` | no | Language code passed to the model |

## Audio Notes

For best results, use clear speech audio with minimal background noise.

Recommended input:

- WAV format
- Single speaker when possible
- 16 kHz sample rate preferred
- Mono preferred

The script can still handle non-16 kHz audio by resampling it to 16 kHz. If the audio has multiple channels, it averages them into a single mono channel before inference.

## How It Works

The script follows this flow:

```text
audio file
  -> load with soundfile
  -> convert to mono if needed
  -> resample to 16 kHz
  -> convert to PyTorch tensor
  -> run AI4Bharat Indic Conformer
  -> print transcription
```

Core code path:

```python
audio = load_audio(args.audio)
audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(device)
transcription = model(audio_tensor, lang=args.lang)
```

## Troubleshooting

If dependencies are missing, reinstall them:

```bash
pip install torch transformers librosa soundfile
```

If model loading is slow, wait for the first download to finish. Later runs should use the cached model.

If CUDA runs out of memory, run on CPU or close other GPU-heavy applications.

If audio loading fails, check that the file path is correct and that the audio format is readable by `soundfile`.

## Roadmap

- [ ] Add `requirements.txt`
- [ ] Add better error messages for missing files and invalid audio
- [ ] Add batch transcription for folders of audio
- [ ] Add transcript export to `.txt` or `.csv`
- [ ] Add WER and CER evaluation scripts
- [ ] Add sample audio instructions
- [ ] Add tests for audio preprocessing
- [ ] Package the script as a cleaner CLI
- [ ] Add optional fine-tuning support

## Suggested GitHub Topics

```text
nepali-asr
automatic-speech-recognition
speech-recognition
nepali-language
conformer
deep-learning
pytorch
ctc
speech-to-text
devanagari
```

## License

No license file has been added yet. Add a license before distributing or reusing this project publicly.

## Author

Created by [Samir Wagle](https://github.com/SamirWagle).
