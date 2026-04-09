# Nepali ASR Conformer

> A focused workspace for building, training, and evaluating Conformer-based automatic speech recognition models for Nepali speech.

[![Project Status](https://img.shields.io/badge/status-scaffold-orange)](#project-status)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#setup)
[![Task](https://img.shields.io/badge/task-automatic%20speech%20recognition-success)](#overview)

## Overview

`nepali-asr-conformer` is intended to become a clean, reproducible ASR pipeline for Nepali speech recognition. The project centers on a Conformer architecture, a modern neural network design that combines convolutional modules with self-attention to capture both local acoustic patterns and long-range speech context.

The goal is to support the full ASR workflow:

- Prepare Nepali speech datasets and transcripts
- Normalize Devanagari text consistently
- Train a Conformer-based acoustic model
- Evaluate recognition quality with WER and CER
- Run inference on Nepali audio files
- Keep experiments reproducible and easy to compare

## Why Conformer for Nepali ASR?

Nepali speech recognition benefits from models that can handle varied speakers, accents, recording conditions, and sentence lengths. Conformer models are a strong fit because they combine:

- **Self-attention** for broader linguistic and acoustic context
- **Convolutions** for local phonetic and spectral patterns
- **Feed-forward layers** for expressive sequence modeling
- **Efficient sequence learning** for end-to-end ASR pipelines

## Project Status

This repository is currently an early scaffold.

The existing codebase contains:

```text
.
|-- .gitattributes
|-- nepali-asr-conformer.py
`-- README.md
```

The main Python file is present as a starting point, but the training, evaluation, and inference pipeline has not been implemented yet.

## Planned Features

- Dataset manifest generation for audio/transcript pairs
- Audio loading, resampling, and feature extraction
- Nepali text normalization for Devanagari transcripts
- Vocabulary or tokenizer preparation
- Conformer encoder model implementation
- CTC-based training loop
- Checkpoint saving and loading
- Evaluation with word error rate and character error rate
- Single-file and batch inference
- Experiment configuration for repeatable training runs

## Setup

Clone the repository:

```bash
git clone https://github.com/SamirWagle/nepali-asr-conformer.git
cd nepali-asr-conformer
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies after a `requirements.txt` or `pyproject.toml` is added:

```bash
pip install -r requirements.txt
```

## Expected Data Format

The pipeline should eventually work with a manifest-style dataset format where each row points to an audio file and its transcript.

Example:

```csv
audio_path,transcript
data/wavs/sample_001.wav,नमस्ते तपाईंलाई कस्तो छ
data/wavs/sample_002.wav,म नेपाली भाषा बोल्छु
```

Recommended transcript conventions:

- Store text as UTF-8
- Normalize Unicode consistently
- Keep a clear policy for punctuation
- Keep a clear policy for numerals
- Avoid mixing Nepali and English text unless code-switching is part of the task
- Track train, validation, and test splits explicitly

## Intended Workflow

Once implemented, the project should follow a workflow like this:

```bash
# 1. Prepare manifests
python nepali-asr-conformer.py prepare-data --data-dir data/raw --out-dir data/manifests

# 2. Train the model
python nepali-asr-conformer.py train --config configs/conformer.yaml

# 3. Evaluate a checkpoint
python nepali-asr-conformer.py evaluate --checkpoint checkpoints/best.pt --manifest data/manifests/test.csv

# 4. Transcribe audio
python nepali-asr-conformer.py transcribe --checkpoint checkpoints/best.pt --audio path/to/audio.wav
```

These commands describe the target interface and may change as the implementation takes shape.

## Evaluation Metrics

The core metrics should be:

- **WER**: Word Error Rate
- **CER**: Character Error Rate

CER is especially useful for Nepali ASR because it gives a more detailed view of Devanagari character-level recognition quality.

## Suggested Repository Structure

As the project grows, a maintainable layout could look like this:

```text
.
|-- configs/
|   `-- conformer.yaml
|-- data/
|   |-- manifests/
|   `-- raw/
|-- notebooks/
|-- src/
|   `-- nepali_asr_conformer/
|       |-- data/
|       |-- models/
|       |-- training/
|       |-- evaluation/
|       `-- inference/
|-- tests/
|-- requirements.txt
`-- README.md
```

## Roadmap

- [ ] Add dependency management
- [ ] Define dataset manifest format
- [ ] Implement Nepali text normalization
- [ ] Add audio preprocessing utilities
- [ ] Implement tokenizer or vocabulary builder
- [ ] Build Conformer model module
- [ ] Add CTC loss training loop
- [ ] Add validation and checkpointing
- [ ] Add WER and CER evaluation
- [ ] Add inference CLI
- [ ] Add tests for preprocessing and metrics

## Contributing

Contributions are welcome once the initial implementation is in place. Useful areas include:

- Nepali text normalization rules
- Dataset preparation scripts
- Model architecture improvements
- Training stability improvements
- Evaluation tooling
- Documentation and examples

Before opening a pull request, keep changes focused and include a short explanation of what changed and how it was tested.

## License

No license file has been added yet. Add a license before distributing or reusing this project publicly.

## Author

Created by [Samir Wagle](https://github.com/SamirWagle).
