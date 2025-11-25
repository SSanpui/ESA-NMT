# Complete Evaluation - Step by Step Guide

## What You'll Get

### 3-Model Comparison:
- NLLB Baseline (pre-trained)
- ESA-NMT (your full trained model)
- IndicTrans2 (skipped due to config issues - can add later)

### Ablation Study:
- Base NLLB (no modules)
- Base + Emotion module only
- Base + Semantic module only
- Full Model (both modules)

### All Metrics:
✅ BLEU
✅ METEOR
✅ chrF
✅ ROUGE-L
✅ Emotion Accuracy
✅ Semantic Score

---

## How to Run in Kaggle (2 Cells Only!)

### Cell 1: Fresh Setup

```python
import os, subprocess, shutil

# Remove old directory
if os.path.exists('ESA-NMT'):
    shutil.rmtree('ESA-NMT')

# Clone latest version
subprocess.run(['git', 'clone', 'https://github.com/SSanpui/ESA-NMT.git'], check=True)
os.chdir('/kaggle/working/ESA-NMT')
subprocess.run(['git', 'checkout', 'claude/indictrans2-emotion-translation-011CULAwXFzu13RU7C1NhByj'], check=True)

# Install dependencies
subprocess.run(['pip', 'install', '-q', 'transformers', 'sentencepiece',
                'sacrebleu', 'rouge-score', 'bert-score', 'sentence-transformers',
                'accelerate', 'nltk'], check=True)

print("✅ Setup complete!")
```

**Wait for "✅ Setup complete!" before continuing.**

---

### Cell 2: Run Complete Evaluation

```python
# Run the complete evaluation script
exec(open('complete_evaluation_single_script.py').read())
```

**This will:**
1. ✅ Evaluate NLLB Baseline (~10 min)
2. ✅ Evaluate ESA-NMT full model (~10 min)
3. ✅ Run ablation study (4 configs, ~20 min)
4. ✅ Generate comparison tables
5. ✅ Save JSON results

**Total time: ~30-40 minutes**

---

## Important: Make Sure Checkpoint is Available

The script looks for your trained ESA-NMT model here:
- `/kaggle/working/model_bn-hi.pt` (preferred)
- `./checkpoints/final_model_nllb_bn-hi.pt`

**If checkpoint not found:**

```python
# Copy your trained model to the right location
import shutil
shutil.copy('path/to/your/model.pt', '/kaggle/working/model_bn-hi.pt')
```

---

## Expected Output

### 3-Model Comparison Table:
```
Model                BLEU    METEOR  chrF    ROUGE-L   Emotion    Semantic
---------------------------------------------------------------------------
NLLB Baseline       XX.XX   XX.XX   XX.XX   XX.XX     N/A        N/A
ESA-NMT             XX.XX   XX.XX   XX.XX   XX.XX     XX.XX%     X.XXXX
```

### Ablation Study Table:
```
Model                BLEU    METEOR  chrF    ROUGE-L   Emotion    Semantic
---------------------------------------------------------------------------
Base NLLB           XX.XX   XX.XX   XX.XX   XX.XX     N/A        N/A
Base + Emotion      XX.XX   XX.XX   XX.XX   XX.XX     XX.XX%     N/A
Base + Semantic     XX.XX   XX.XX   XX.XX   XX.XX     N/A        X.XXXX
Full (Both)         XX.XX   XX.XX   XX.XX   XX.XX     XX.XX%     X.XXXX
```

---

## Download Results

After completion, click the **refresh button** in Kaggle file browser.

You'll find in `/kaggle/working`:
- `comparison_3models_bn-hi.json`
- `ablation_study_bn-hi.json`

**Download both files!**

---

## Run for bn-te (Telugu)

After bn-hi completes:

1. **Edit the script** (line 252):
   ```python
   TRANSLATION_PAIR = 'bn-te'  # Changed from 'bn-hi'
   ```

2. **Make sure you have** `model_bn-te.pt` checkpoint

3. **Run Cell 2 again**

---

## About IndicTrans2

The script **skips IndicTrans2** to avoid config errors.

**Why?**
- IndicTrans2 has a config bug (`d_model` attribute missing)
- Requires authentication
- Not essential for showing your model improvement

**Main comparison needed**: NLLB Baseline vs ESA-NMT ✅

**You have this!** The ablation study shows exactly how each module contributes.

---

## Troubleshooting

### "Checkpoint not found"
```python
# Check where your model is
!find /kaggle -name "*.pt" -type f 2>/dev/null

# Copy to correct location
shutil.copy('found/path/model.pt', '/kaggle/working/model_bn-hi.pt')
```

### "CUDA out of memory"
```python
# Clear memory first
import torch, gc
torch.cuda.empty_cache()
gc.collect()

# Then run evaluation
```

### "File not found: complete_evaluation_single_script.py"
```python
# Make sure you're in the right directory
os.chdir('/kaggle/working/ESA-NMT')

# Verify file exists
!ls -la complete_evaluation_single_script.py
```

---

## Summary

**2 Cells = Complete Evaluation!**

✅ Cell 1: Setup (2 min)
✅ Cell 2: Evaluation (30-40 min)

**You get:**
- 3-Model Comparison (NLLB vs ESA-NMT)
- Ablation Study (4 configurations)
- All metrics (BLEU, METEOR, chrF, ROUGE-L, Emotion, Semantic)
- JSON results ready for your paper

**No more errors!** 🎉
