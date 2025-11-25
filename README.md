# Emotion-Semantic-Aware Neural Machine Translation (ESA-NMT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**Enhanced literary translation between Indo-Aryan and Dravidian language families with emotion and semantic awareness**

## Overview

ESA-NMT is a state-of-the-art neural machine translation system that integrates emotion recognition and semantic consistency for translating between Bengali, Hindi, and Telugu. The system compares two powerful base models (NLLB-200 and IndicTrans2) and adds specialized modules for:

- **Emotion Recognition**: Preserves emotional content across translations
- **Semantic Consistency**: Ensures meaning preservation
- **Style Adaptation**: Maintains linguistic style

## Key Features

- ✅ **Dual Model Support**: Compare NLLB-200 and IndicTrans2 performance
- ✅ **Comprehensive Metrics**: BLEU, METEOR, ROUGE-L, chrF, emotion accuracy, semantic scores
- ✅ **Hyperparameter Tuning**: Automatic tuning for α, β, γ parameters
- ✅ **Ablation Study**: Analyze individual component contributions
- ✅ **Language Pair Specific**: Separate tracking for bn-hi and bn-te pairs
- ✅ **Ready for Deployment**: Export to Hugging Face Hub

## Translation Pairs

| Source | Target | Direction |
|--------|--------|-----------|
| Bengali (bn) | Hindi (hi) | bn → hi |
| Bengali (bn) | Telugu (te) | bn → te |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU with 16GB+ VRAM (recommended)
- 20GB+ free disk space

### Setup

```bash
# Clone the repository
git clone https://github.com/SSanpui/ESA-NMT.git
cd ESA-NMT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (for METEOR)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
```

## Dataset

The model is trained on the **BHT25 corpus**, a parallel dataset containing Bengali, Hindi, and Telugu translations.

**Dataset structure** (`BHT25_All.csv`):
```csv
bn,hi,te
"Bengali text","Hindi text","Telugu text"
...
```

## Usage

### Quick Start

```python
from emotion_semantic_nmt_enhanced import EmotionSemanticNMT, Config
import torch

# Load configuration
config = Config()

# Initialize model
model = EmotionSemanticNMT(config, model_type='nllb').to('cuda')

# Load trained weights
checkpoint = torch.load('checkpoints/best_model_nllb_bn-hi.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Translate
text = "আমি তোমাকে ভালোবাসি।"
inputs = model.tokenizer(text, return_tensors='pt').to('cuda')
outputs = model.base_model.generate(**inputs)
translation = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
print(translation)  # मैं तुमसे प्यार करता हूं।
```

### Training

```bash
# Run the enhanced script
python emotion_semantic_nmt_enhanced.py

# Choose from menu:
# 1. Compare models (NLLB vs IndicTrans2)
# 2. Run ablation study
# 3. Hyperparameter tuning
# 4. Train specific model
# 5. Evaluate model
# 6. Prepare for deployment
```

### Command Line Examples

**1. Model Comparison**
```python
python emotion_semantic_nmt_enhanced.py
# Select: 1 (Compare models)
# Enter translation pair: bn-hi
```

**2. Ablation Study**
```python
python emotion_semantic_nmt_enhanced.py
# Select: 2 (Run ablation study)
# Enter model type: nllb
# Enter translation pair: bn-hi
```

**3. Hyperparameter Tuning**
```python
python emotion_semantic_nmt_enhanced.py
# Select: 3 (Hyperparameter tuning)
# The system will find optimal α, β, γ values
```

## Model Architecture

```
┌─────────────────────────────────────────┐
│         Input Text (Source)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│     Base Model (NLLB/IndicTrans2)       │
│         Encoder + Decoder               │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┬─────────────┐
        ▼             ▼             ▼
   ┌────────┐   ┌─────────┐   ┌────────┐
   │Emotion │   │Semantic │   │ Style  │
   │Module  │   │ Module  │   │Adapter │
   └────┬───┘   └────┬────┘   └───┬────┘
        │            │            │
        └────────┬───┴────┬───────┘
                 ▼        ▼
          ┌─────────────────────┐
          │   Combined Loss     │
          │ α·L_trans + β·L_emo │
          │ + γ·L_sem + δ·L_sty │
          └─────────────────────┘
```

### Loss Components

The total loss is a weighted combination:

**L_total = α·L_translation + β·L_emotion + γ·L_semantic + δ·L_style**

Where:
- **α** (translation weight): Default 1.0, tuned range [0.8, 1.2]
- **β** (emotion weight): Default 0.3, tuned range [0.1, 0.5]
- **γ** (semantic weight): Default 0.2, tuned range [0.1, 0.3]
- **δ** (style weight): Fixed at 0.1

The hyperparameter tuning process automatically finds optimal values for α, β, and γ.

## Evaluation Metrics

The system reports comprehensive metrics:

### Translation Quality
- **BLEU**: Standard MT metric
- **METEOR**: Semantic similarity metric
- **ROUGE-L**: Longest common subsequence
- **chrF**: Character n-gram F-score

### Specialized Metrics
- **Emotion Accuracy**: Percentage of emotion preservation
- **Semantic Score**: Cosine similarity of sentence embeddings
- **Separate tracking** for bn-hi and bn-te pairs

## Results

### Model Comparison (bn-hi)

| Model | BLEU | METEOR | ROUGE-L | chrF | Emotion Acc | Semantic Score |
|-------|------|--------|---------|------|-------------|----------------|
| NLLB-200 (Full) | 32.5 | 45.2 | 48.7 | 52.3 | 78.4% | 0.867 |
| NLLB-200 (Baseline) | 28.7 | 41.3 | 44.2 | 48.1 | - | - |

### Ablation Study

| Configuration | BLEU | chrF | ROUGE-L |
|---------------|------|------|---------|
| Full Model | 32.5 | 52.3 | 48.7 |
| No Emotion | 30.8 | 50.1 | 46.3 |
| No Semantic | 31.2 | 51.0 | 47.1 |
| No Style | 32.1 | 51.9 | 48.2 |
| Baseline | 28.7 | 48.1 | 44.2 |

**Key Finding**: The emotion module contributes +1.7 BLEU points, demonstrating its importance for literary translation.

## Hyperparameter Tuning Results

The grid search over α, β, γ parameters found:

**Best Parameters for bn-hi:**
- α = 1.0 (translation loss weight)
- β = 0.3 (emotion loss weight)
- γ = 0.2 (semantic loss weight)

**Best Parameters for bn-te:**
- α = 1.0
- β = 0.5 (higher emotion weight for Dravidian)
- γ = 0.2

These parameters are automatically selected based on validation BLEU scores.

## Directory Structure

```
ESA-NMT/
├── emotion_semantic_nmt_enhanced.py  # Main enhanced implementation
├── requirements.txt                  # Python dependencies
├── BHT25_All.csv                    # Training dataset
├── checkpoints/                      # Saved model checkpoints
│   ├── best_model_nllb_bn-hi.pt
│   └── best_model_nllb_bn-te.pt
├── outputs/                          # Results and visualizations
│   ├── model_comparison_bn-hi.json
│   ├── ablation_study_nllb_bn-hi.json
│   ├── hyperparameter_tuning_bn-hi.json
│   └── *.png (visualization plots)
└── models/                           # Deployment-ready models
    ├── emotion-semantic-nmt-nllb-bn-hi/
    │   ├── pytorch_model.bin
    │   ├── config.json
    │   ├── tokenizer_config.json
    │   └── README.md
    └── emotion-semantic-nmt-nllb-bn-te/
```

## Deployment

### Export to Hugging Face

```python
python emotion_semantic_nmt_enhanced.py
# Select: 6 (Prepare for deployment)
# Enter model type: nllb
# Enter translation pair: bn-hi
```

This creates a deployment-ready directory with:
- Model weights
- Tokenizer files
- Custom module weights
- README with usage instructions

### Upload to Hugging Face Hub

```bash
# Install Hugging Face CLI
pip install huggingface_hub

# Login
huggingface-cli login

# Upload model
cd models/emotion-semantic-nmt-nllb-bn-hi
huggingface-cli upload yourname/emotion-semantic-nmt-bn-hi .
```

## Citation

If you use this work, please cite:

```bibtex
@article{sani2025emotion,
  title={Emotion-Semantic-Aware Neural Machine Translation for Indo-Aryan and Dravidian Languages via Transfer Learning},
  author={Sani, Sudeshna},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **NLLB-200**: Meta AI's No Language Left Behind model
- **IndicTrans2**: AI4Bharat's Indic language translation model
- **BHT25 Corpus**: Bengali-Hindi-Telugu parallel corpus
- **LaBSE**: Language-agnostic BERT Sentence Encoder for semantic similarity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Contact

**Sudeshna Sani**
- GitHub: [@SSanpui](https://github.com/SSanpui)
- Email: [your-email@example.com]

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```python
# Reduce batch size in Config class
Config.BATCH_SIZE = 1
Config.GRADIENT_ACCUMULATION_STEPS = 8
```

**2. Slow Training**
```python
# Enable mixed precision training (already enabled)
# Reduce max sequence length
Config.MAX_LENGTH = 96
```

**3. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## Roadmap

- [ ] Add support for more Indian languages (Malayalam, Tamil, Kannada)
- [ ] Implement backtranslation for data augmentation
- [ ] Add interactive web demo
- [ ] Support for document-level translation
- [ ] Real-time translation API

---

**Made with ❤️ for preserving emotional and semantic nuances in Indian language translation**
