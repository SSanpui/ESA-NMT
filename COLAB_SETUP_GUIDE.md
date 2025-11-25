# ESA-NMT Google Colab Setup Guide

## Overview

This guide walks you through running ESA-NMT training and deployment entirely in Google Colab - no local setup required!

## Prerequisites

1. **Google Account** - For accessing Google Colab
2. **Hugging Face Account** (for deployment) - Create at [huggingface.co/join](https://huggingface.co/join)
3. **GitHub Account** (optional) - For accessing the repository

## What You Need to Upload to GitHub

To run the notebook, your GitHub repository should contain:

### Required Files

```
ESA-NMT/
├── ESA_NMT_Research.ipynb          ✅ Main notebook (already created)
├── train_esa_nmt.py                 ✅ Training script
├── evaluate_esa_nmt.py              ✅ Evaluation script
├── deploy_to_hf.py                  ✅ HF deployment script
├── requirements.txt                 ✅ Dependencies
├── README.md                        ✅ Documentation
└── scripts/
    └── annotate_emotions.py         ✅ Annotation script
```

### Dataset Files

**Option 1: Use Hugging Face Dataset (Recommended)**
- Upload BHT25 to Hugging Face Datasets: [huggingface.co/new-dataset](https://huggingface.co/new-dataset)
- The notebook will download it automatically

**Option 2: Include in Repository**
- Add `BHT25_All.csv` to your repository root
- The notebook will find it automatically

## Step-by-Step Guide

### Step 1: Upload Files to GitHub

1. Go to [github.com](https://github.com) and create a new repository named `ESA-NMT`
2. Upload all the files listed above:
   - Click "Add file" → "Upload files"
   - Drag and drop all Python files
   - Commit the changes

3. Upload the notebook:
   - Upload `ESA_NMT_Research.ipynb` to the repository root

### Step 2: Open Notebook in Colab

**Method 1: Direct Link**
1. Go to your notebook on GitHub
2. Copy the URL
3. Open [colab.research.google.com](https://colab.research.google.com)
4. Click "GitHub" tab
5. Paste your repository URL
6. Select the notebook

**Method 2: Colab Badge**
The notebook has a "Open in Colab" badge at the top. Just click it!

### Step 3: Enable GPU

**Critical Step - Don't Skip!**

1. In Colab, go to: **Runtime → Change runtime type**
2. Hardware accelerator: Select **GPU**
3. GPU type: 
   - Free tier: **T4** (16GB) - Total training ~12-15 hours
   - Colab Pro: **V100** (16GB) - Total training ~6-8 hours
   - Colab Pro+: **A100** (40GB) - Total training ~3-4 hours
4. Click **Save**

### Step 4: Configure Experiment

Run the first code cell to set your configuration:

```python
RUN_MODE = "full_training"  # Options: "quick_demo", "full_training", "ablation"
TRANSLATION_PAIR = "bn-hi"  # Options: "bn-hi", "bn-te"
```

**Run Modes Explained:**

| Mode | Duration | Purpose |
|------|----------|---------|
| `quick_demo` | 30-45 min | Test pipeline with 500 samples |
| `full_training` | 6-15 hours | Complete 3-phase training (recommended) |
| `ablation` | 8-12 hours | All configurations for ablation study |
| `weight_tuning` | 6-10 hours | Test different loss weights |

### Step 5: Run the Notebook

**For Quick Demo:**
1. Run cells sequentially from top to bottom
2. Wait for each cell to complete before running the next
3. The notebook will automatically skip unnecessary steps

**For Full Training:**
1. Make sure you have enough time (6-15 hours depending on GPU)
2. Run all cells sequentially
3. Consider using the keep-alive script (see below) to prevent disconnection

**Keep-Alive Script (Important for Long Training):**

Colab may disconnect if the tab is inactive. To prevent this:

1. Open browser console: Press `F12`
2. Paste this code:

```javascript
function KeepAlive(){
  console.log("Keep-alive: " + new Date().toTimeString());
  document.querySelector("colab-connect-button").shadowRoot.querySelector("#connect").click();
}
setInterval(KeepAlive, 60000);
```

3. Press Enter

### Step 6: Download Results

After training completes:

1. Run the "Download Results" cells at the end
2. A file `esa_nmt_results.zip` will be created
3. Click the download button or run the download cell
4. Extract the ZIP file to access:
   - Trained model checkpoints
   - Evaluation metrics (JSON files)
   - Visualizations (PNG files)

### Step 7: Deploy to Hugging Face (Optional)

**Prerequisites:**
1. Create Hugging Face account at [huggingface.co/join](https://huggingface.co/join)
2. Get your access token:
   - Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Name it "ESA-NMT"
   - Select "Write" permission
   - Copy the token

**In Colab:**

```python
# Install HF Hub
!pip install huggingface_hub

# Login (paste your token when prompted)
!huggingface-cli login

# Deploy
!python deploy_to_hf.py \
    --model_dir ./outputs/phase3_joint \
    --repo_name ESA-NMT-bn-hi \
    --hf_username YOUR_USERNAME \
    --translation_pair bn-hi \
    --metrics_file ./outputs/evaluation/results.json
```

Replace `YOUR_USERNAME` with your Hugging Face username.

Your model will be available at: `https://huggingface.co/YOUR_USERNAME/ESA-NMT-bn-hi`

## Expected Outputs

### During Training

You'll see progress bars showing:
- Epoch progress
- Current loss values
- Training speed (samples/sec)
- Estimated time remaining

Example output:
```
Phase 1: Emotion Module Pre-training (3 epochs)
Epoch 1/3: 100%|██████████| 4749/4749 [1:23:45<00:00, 0.95s/it, loss=2.341]
Epoch 1 - Average Training Loss: 2.3412
Epoch 1 - Validation Loss: 2.1245
Saved best model to ./outputs/phase1_emotion/phase1_best_model.pt
```

### Results Files

After completion, you'll have:

```
outputs/
├── phase1_emotion/
│   ├── phase1_best_model.pt
│   └── config.json
├── phase2_semantic/
│   ├── phase2_best_model.pt
│   └── config.json
├── phase3_joint/
│   ├── final_model.pt
│   ├── phase3_best_model.pt
│   ├── tokenizer files...
│   └── config.json
└── evaluation/
    ├── results.json
    ├── predictions.csv
    ├── translation_metrics.png
    └── emotion_confusion_matrix.png
```

### Sample Results (Expected Values)

**Bengali-Hindi:**
```json
{
  "translation_metrics": {
    "bleu": 42.66,
    "meteor": 63.04,
    "rouge_l": 0.818,
    "chrf": 62.60
  },
  "emotion_metrics": {
    "accuracy": 76.57
  },
  "semantic_metrics": {
    "mean_similarity": 0.9290
  }
}
```

## Troubleshooting

### Problem: "No GPU detected"

**Solution:**
1. Runtime → Change runtime type → GPU
2. If already set, try: Runtime → Factory reset runtime
3. Wait a few minutes and try again

### Problem: "Out of Memory"

**Solutions:**
```python
# In configuration cell, reduce batch size:
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # Increase this

# Or add these flags to training:
--gradient_checkpointing
--mixed_precision fp16
```

### Problem: Colab Disconnected During Training

**Prevention:**
- Use the keep-alive JavaScript (see Step 5)
- Consider upgrading to Colab Pro for longer runtimes
- Save checkpoints frequently (already enabled in notebook)

**Recovery:**
- Reopen the notebook
- The training will resume from last checkpoint automatically

### Problem: Dataset Not Found

**Solutions:**

**If using Hugging Face:**
```python
# Install datasets library
!pip install datasets

# Download dataset
from datasets import load_dataset
dataset = load_dataset("SSanpui/BHT25")
dataset.to_csv("BHT25_All.csv")
```

**If using local file:**
- Upload `BHT25_All.csv` to Colab:
  1. Click folder icon on left sidebar
  2. Click upload button
  3. Select your CSV file

### Problem: Model Too Large to Download

**Solution:**
```python
# Download in parts
!zip -s 100m esa_nmt_results.zip --out split_results.zip

# Or upload directly to Google Drive:
from google.colab import drive
drive.mount('/content/drive')
!cp -r ./outputs /content/drive/MyDrive/ESA-NMT-outputs/
```

## Time Estimates by GPU Type

| GPU Type | Quick Demo | Full Training | Ablation Study |
|----------|------------|---------------|----------------|
| T4 (Free) | 30-45 min | 12-15 hours | 15-18 hours |
| V100 (Pro) | 15-20 min | 6-8 hours | 8-10 hours |
| A100 (Pro+) | 10-15 min | 3-4 hours | 4-6 hours |

## Cost Considerations

**Free Tier:**
- 12-15 hours may require multiple sessions
- Sessions limited to ~12 hours
- Use checkpointing to resume

**Colab Pro ($9.99/month):**
- V100 GPU
- Longer sessions (~24 hours)
- Can complete training in one session

**Colab Pro+ ($49.99/month):**
- A100 GPU (fastest)
- Background execution
- Ideal for research

## Best Practices

1. **Start with Quick Demo**
   - Test the pipeline first (30-45 minutes)
   - Verify everything works
   - Then run full training

2. **Monitor Progress**
   - Check loss values are decreasing
   - Watch for OOM errors
   - Keep browser tab visible (prevent sleep)

3. **Save Frequently**
   - The notebook saves checkpoints automatically
   - Download results periodically
   - Back up to Google Drive

4. **Use Version Control**
   - Commit changes to GitHub regularly
   - Tag important versions
   - Document experiments

## Next Steps After Training

1. **Evaluate Model**
   - Review metrics in `results.json`
   - Check visualizations
   - Compare with baseline (Base NLLB)

2. **Deploy Model**
   - Upload to Hugging Face Hub
   - Share with community
   - Create model card

3. **Use for Inference**
   - Download model locally
   - Or use from Hugging Face
   - Integrate into applications

## Support

If you encounter issues:

1. Check this guide first
2. Review the main [README.md](README.md)
3. Open an issue on GitHub
4. Include:
   - Error message
   - Cell that failed
   - GPU type
   - Configuration used

## Summary Checklist

- [ ] GitHub repository created with all files
- [ ] Dataset uploaded (HF or GitHub)
- [ ] Notebook opened in Colab
- [ ] GPU enabled (T4/V100/A100)
- [ ] Configuration set (run mode, translation pair)
- [ ] Keep-alive script running (for long training)
- [ ] Training started
- [ ] Results downloaded
- [ ] Model deployed to Hugging Face (optional)

**Ready to start? Open the notebook and begin training!**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/ESA-NMT/blob/main/ESA_NMT_Research.ipynb)

Replace `YOUR_USERNAME` with your GitHub username in the link above.
