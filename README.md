# ESA-NMT: Emotion-Semantic-Aware Neural Machine Translation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SSanpui/ESA-NMT/blob/main/ESA_NMT_Research.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Multi-task neural machine translation system for cross-family Indian languages with emotion preservation and semantic consistency.

## Overview

ESA-NMT addresses the challenge of translating literary content across linguistically diverse Indian languages while preserving emotional nuances and semantic meaning. The system combines:

- **Translation**: NLLB-200 base model for Bengali-Hindi-Telugu translation
- **Emotion Preservation**: XLM-RoBERTa-based classifier for 8 emotions (Plutchik's wheel)
- **Semantic Consistency**: LaBSE-based module ensuring meaning preservation

### Key Results

| Metric | Bengali-Hindi | Bengali-Telugu |
|--------|---------------|----------------|
| BLEU | 42.66 | 36.74 |
| METEOR | 63.04 | 51.40 |
| ROUGE-L | 0.818 | 0.904 |
| chrF | 62.60 | 61.58 |
| Emotion Accuracy | 76.57% | 77.90% |
| Semantic Similarity | 0.9290 | 0.9185 |

**Overall Performance:**
- 8-Emotion Classification: 77.2% accuracy
- Average Semantic Consistency: 0.92 cosine similarity
- GPU Memory Reduction: 35% vs standard training

## Quick Start with Google Colab

The easiest way to run ESA-NMT is using Google Colab:

1. Click the "Open in Colab" badge above
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells sequentially
4. Download results when complete

**No local setup required!**

## Repository Structure

```
ESA-NMT/
├── ESA_NMT_Research.ipynb      # Main Colab notebook
├── train_esa_nmt.py             # Training script
├── evaluate_esa_nmt.py          # Evaluation script
├── deploy_to_hf.py              # Hugging Face deployment
├── scripts/
│   └── annotate_emotions.py     # Emotion annotation script
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── LICENSE                      # Apache 2.0 license
```

## Dataset: BHT25

**BHT25** is a curated parallel corpus of 25,000 literary text samples across Bengali, Hindi, and Telugu.

- **Access**: [SSanpui/BHT25 on Hugging Face](https://huggingface.co/datasets/SSanpui/BHT25)
- **Domain**: Literary content (novels, stories, poetry)
- **Annotations**: 8-emotion labels + semantic similarity scores
- **Split**: 18,995 train / 4,070 validation / 4,070 test

### Download Dataset

The notebook automatically handles dataset loading. For manual download:

```python
from datasets import load_dataset
dataset = load_dataset("SSanpui/BHT25")
```

Or download CSV directly from Hugging Face.

## Installation (Local Setup)

If you want to run locally instead of Colab:

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (16GB+ recommended)
- 50GB free disk space

### Setup

```bash
# Clone repository
git clone https://github.com/SSanpui/ESA-NMT.git
cd ESA-NMT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Usage

### 1. Emotion Annotation (One-time)

Annotate the BHT25 dataset with emotion labels and semantic scores:

```bash
python scripts/annotate_emotions.py \
    --input_file BHT25_All.csv \
    --output_file BHT25_annotated.csv \
    --emotion_model xlm-roberta-base \
    --semantic_model sentence-transformers/LaBSE \
    --batch_size 32
```

**Time**: ~45-60 minutes for 25,000 samples

### 2. Training

#### Quick Demo (30-45 minutes)

```bash
python train_esa_nmt.py \
    --translation_pair bn-hi \
    --base_model facebook/nllb-200-distilled-600M \
    --emotion_model xlm-roberta-base \
    --semantic_model sentence-transformers/LaBSE \
    --max_samples 500 \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --output_dir ./outputs/quick_demo \
    --skip_progressive_training
```

#### Full Progressive Training (6-8 hours on V100)

**Phase 1: Emotion Module Pre-training**
```bash
python train_esa_nmt.py \
    --translation_pair bn-hi \
    --phase 1 \
    --num_epochs 3 \
    --freeze_base_model \
    --output_dir ./outputs/phase1_emotion
```

**Phase 2: Semantic Module Pre-training**
```bash
python train_esa_nmt.py \
    --translation_pair bn-hi \
    --phase 2 \
    --num_epochs 3 \
    --freeze_base_model \
    --load_emotion_module ./outputs/phase1_emotion/best_emotion_module.pt \
    --output_dir ./outputs/phase2_semantic
```

**Phase 3: Joint Multi-Task Fine-tuning**
```bash
python train_esa_nmt.py \
    --translation_pair bn-hi \
    --phase 3 \
    --num_epochs 9 \
    --alpha 1.0 \
    --beta 0.3 \
    --gamma 0.2 \
    --load_emotion_module ./outputs/phase1_emotion/best_emotion_module.pt \
    --load_semantic_module ./outputs/phase2_semantic/best_semantic_module.pt \
    --gradient_checkpointing \
    --mixed_precision fp16 \
    --output_dir ./outputs/phase3_joint
```

### 3. Evaluation

```bash
python evaluate_esa_nmt.py \
    --translation_pair bn-hi \
    --model_dir ./outputs/phase3_joint \
    --base_model facebook/nllb-200-distilled-600M \
    --compute_all_metrics \
    --save_predictions \
    --output_dir ./outputs/evaluation
```

### 4. Deploy to Hugging Face

```bash
# Login to Hugging Face (first time only)
pip install huggingface_hub
huggingface-cli login

# Deploy model
python deploy_to_hf.py \
    --model_dir ./outputs/phase3_joint \
    --repo_name ESA-NMT-bn-hi \
    --hf_username YOUR_USERNAME \
    --translation_pair bn-hi \
    --metrics_file ./outputs/evaluation/results.json
```

Your model will be available at: `https://huggingface.co/YOUR_USERNAME/ESA-NMT-bn-hi`

### 5. Inference

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model
model_name = "YOUR_USERNAME/ESA-NMT-bn-hi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Translate
bengali_text = "আমি তোমাকে ভালোবাসি"
inputs = tokenizer(bengali_text, return_tensors="pt", src_lang="ben_Beng")
outputs = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["hin_Deva"], max_length=128, num_beams=5)
hindi_translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(hindi_translation)
```

## Training Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | NLLB-200-600M | Foundation translation model |
| Emotion Model | XLM-RoBERTa-base | Cross-lingual emotion classifier |
| Semantic Model | LaBSE | Sentence embedding model |
| Batch Size | 1 | Per-device batch size |
| Gradient Accumulation | 4 | Effective batch size = 4 |
| Learning Rate | 2e-5 | AdamW optimizer |
| Total Epochs | 9 | Phase 3 joint training |
| Loss Weights (α,β,γ) | (1.0, 0.3, 0.2) | Translation, emotion, semantic |
| Mixed Precision | FP16 | Memory optimization |

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA T4 (16GB)
- RAM: 16GB
- Storage: 50GB

**Recommended:**
- GPU: NVIDIA V100 (16GB) or A100 (40GB)
- RAM: 32GB
- Storage: 100GB

**Training Time:**
- T4: 12-15 hours (full training)
- V100: 6-8 hours (full training)
- A100: 3-4 hours (full training)

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    ESA-NMT Architecture                  │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input (Bengali) → [Tokenizer]                          │
│                         ↓                                │
│                   [NLLB-200 Encoder]                     │
│                    /     |      \                        │
│                   /      |       \                       │
│           [Translation] [Emotion] [Semantic]             │
│              Module     Module    Module                 │
│                  \       |       /                       │
│                   \      |      /                        │
│                [Multi-Objective Loss]                    │
│          L = α·L_trans + β·L_emo + γ·L_sem              │
│                         ↓                                │
│                   [NLLB-200 Decoder]                     │
│                         ↓                                │
│                 Output (Hindi/Telugu)                    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Emotion Categories (Plutchik's Wheel)

1. **Joy** - happiness, contentment, pleasure
2. **Sadness** - sorrow, grief, melancholy
3. **Anger** - fury, resentment, irritation
4. **Fear** - anxiety, apprehension, terror
5. **Surprise** - astonishment, amazement, wonder
6. **Trust** - acceptance, confidence, belief
7. **Disgust** - aversion, revulsion, contempt
8. **Anticipation** - expectation, interest, vigilance

## Ablation Study

Systematic evaluation of module contributions:

| Configuration | BLEU | METEOR | ROUGE-L | chrF | Emotion | Semantic |
|--------------|------|--------|---------|------|---------|----------|
| Base NLLB | 27.6 | 36.8 | 0.354 | 51.2 | 61.30% | 0.8920 |
| + Emotion | 38.24 | 58.12 | 0.645 | 58.76 | 73.48% | 0.9015 |
| + Semantic | 39.15 | 59.84 | 0.712 | 60.32 | 65.12% | 0.9185 |
| **Full ESA-NMT** | **42.66** | **63.04** | **0.818** | **62.60** | **76.57%** | **0.9290** |

## Citation

If you use ESA-NMT or the BHT25 dataset in your research, please cite:

```bibtex
@article{sanpui2024esanmt,
  title={ESA-NMT: Emotion-Semantic-Aware Neural Machine Translation for Cross-Family Indian Languages},
  author={Sanpui, Sudeshna},
  journal={IEEE Access},
  year={2024},
  note={Accepted for publication}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
--batch_size 1

# Increase gradient accumulation
--gradient_accumulation_steps 8

# Enable gradient checkpointing
--gradient_checkpointing

# Use mixed precision
--mixed_precision fp16
```

### Slow Training

- Upgrade to V100/A100 GPU
- Enable mixed precision training
- Use compiled model (PyTorch 2.0+)

### Low Performance

- Verify emotion annotations are correct
- Ensure all three training phases completed
- Check loss weight configuration
- Validate dataset quality

## Acknowledgments

- Facebook AI Research for NLLB-200
- Hugging Face for the Transformers library
- Google for LaBSE sentence embeddings
- XLM-RoBERTa team for cross-lingual representations

## Contact

- **Author**: Sudeshna Sanpui
- **GitHub**: [@SSanpui](https://github.com/SSanpui)
- **Dataset**: [SSanpui/BHT25](https://huggingface.co/datasets/SSanpui/BHT25)
- **Issues**: [GitHub Issues](https://github.com/SSanpui/ESA-NMT/issues)

---

**Note**: This is research software. For production use, additional testing and optimization are recommended.
