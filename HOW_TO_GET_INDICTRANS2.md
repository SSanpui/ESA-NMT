# How to Get IndicTrans2 Working

## Issue: Auth Required

IndicTrans2 is a "gated" model - requires Hugging Face account and token.

## Step-by-Step Fix (5 minutes)

### 1. Create Hugging Face Account
- Go to https://huggingface.co/join
- Sign up (free)

### 2. Get Access Token
- Go to https://huggingface.co/settings/tokens
- Click **"New token"**
- Name: `kaggle_indictrans2`
- Type: **Read**
- Click **"Generate"**
- **Copy the token** (starts with `hf_...`)

### 3. Request Model Access
- Go to https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
- Click **"Request access"** button
- Fill form (takes 1 minute)
- **Wait for approval** (usually instant, max 24 hours)

### 4. Use Token in Kaggle

When you run the script, it will ask:
```
Enter your Hugging Face token (or Enter to skip):
```

**Paste your token** (hf_xxxxx) and press Enter.

## Alternative: Skip IndicTrans2

If you don't want to wait for approval:
- Just press **Enter** when asked for token
- Script will evaluate only NLLB Baseline vs ESA-NMT
- You can add IndicTrans2 later

## What You Get

### With IndicTrans2:
```
Model                     BLEU    METEOR  chrF    ROUGE-L   Emotion    Semantic
---------------------------------------------------------------------------------
NLLB Baseline            XX.XX   XX.XX   XX.XX   XX.XX     N/A        N/A
ESA-NMT (Proposed)       XX.XX   XX.XX   XX.XX   XX.XX     XX.XX%     X.XXXX
IndicTrans2              XX.XX   XX.XX   XX.XX   XX.XX     N/A        N/A
```

### Without IndicTrans2:
```
Model                     BLEU    METEOR  chrF    ROUGE-L   Emotion    Semantic
---------------------------------------------------------------------------------
NLLB Baseline            XX.XX   XX.XX   XX.XX   XX.XX     N/A        N/A
ESA-NMT (Proposed)       XX.XX   XX.XX   XX.XX   XX.XX     XX.XX%     X.XXXX
```

## Timeline

- **With token**: 30 minutes (all 3 models)
- **Without token**: 20 minutes (2 models)

## METEOR Score

✅ Now included in all evaluations!
- Downloads nltk data automatically
- Shows in comparison table
- Saved in JSON results

## Quick Start

```python
# In Kaggle, just run:
exec(open('complete_3model_comparison_with_meteor.py').read())

# When prompted:
# - Paste your HF token (if you have one)
# - Or press Enter to skip IndicTrans2
```

That's it! 🚀
