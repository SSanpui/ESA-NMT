#!/usr/bin/env python3
"""
Retrain ESA-NMT with properly integrated emotion/semantic modules
Fixed to handle hi-te pair and PyTorch compatibility
"""

import torch
import gc
import os
import sys
import pandas as pd
import json
from datetime import datetime

# Import from local modules
from src.emotion_semantic_nmt_enhanced import (
    Config, EmotionSemanticNMT, Trainer, ComprehensiveEvaluator,
    device, config
)
from src.dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

print("""
╔═══════════════════════════════════════════════════════════════╗
║  RETRAIN ESA-NMT with Fixed Code (GAMMA=0.5)                 ║
║  Supports: bn-hi, bn-te, hi-te                               ║
╚═══════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# CONFIGURATION
# ============================================================================
TRANSLATION_PAIR = 'hi-te'  # Options: 'bn-hi', 'bn-te', 'hi-te'
CSV_PATH = 'BHT25_All_annotated.csv'

print(f"\n📋 Configuration:")
print(f"   Translation pair: {TRANSLATION_PAIR}")
print(f"   Source language: {TRANSLATION_PAIR.split('-')[0]}")
print(f"   Target language: {TRANSLATION_PAIR.split('-')[1]}")
print(f"   Loss weights: α={config.ALPHA}, β={config.BETA}, γ={config.GAMMA}, δ={config.DELTA}")
print(f"   Epochs: {config.EPOCHS['phase1']}")
print(f"   Batch size: {config.BATCH_SIZE}")
print(f"   Device: {device}")

# Validate language pair
valid_pairs = ['bn-hi', 'bn-te', 'hi-te']
if TRANSLATION_PAIR not in valid_pairs:
    print(f"\n❌ ERROR: Invalid translation pair '{TRANSLATION_PAIR}'")
    print(f"   Valid options: {valid_pairs}")
    exit(1)

# ============================================================================
# CHECK DATASET
# ============================================================================
print(f"\n🔍 Checking dataset for {TRANSLATION_PAIR}...")

if not os.path.exists(CSV_PATH):
    if os.path.exists(f'data/{CSV_PATH}'):
        CSV_PATH = f'data/{CSV_PATH}'
        print(f"✅ Found dataset at: {CSV_PATH}")
    else:
        print(f"❌ Dataset not found")
        exit(1)

df = pd.read_csv(CSV_PATH)
print(f"✅ Dataset loaded with {len(df)} rows")

src_lang, tgt_lang = TRANSLATION_PAIR.split('-')

# Check columns
required_text = [src_lang, tgt_lang]
missing_text = [col for col in required_text if col not in df.columns]

if missing_text:
    print(f"❌ ERROR: Missing text columns: {missing_text}")
    exit(1)

print(f"✅ Text columns found: {src_lang}, {tgt_lang}")

# Count valid rows
valid_rows = df.dropna(subset=[src_lang, tgt_lang])
print(f"📊 Valid parallel sentences for {TRANSLATION_PAIR}: {len(valid_rows)}")

# ============================================================================
# CREATE MODEL
# ============================================================================
print(f"\n1️⃣ Creating ESA-NMT model for {TRANSLATION_PAIR}...")

def get_nllb_lang_code(lang):
    """Convert language code to NLLB format"""
    lang_map = {
        'bn': 'ben_Beng',
        'hi': 'hin_Deva', 
        'te': 'tel_Telu'
    }
    return lang_map.get(lang, f'{lang}_Latn')

src_nllb = get_nllb_lang_code(src_lang)
tgt_nllb = get_nllb_lang_code(tgt_lang)

print(f"   NLLB source code: {src_nllb}")
print(f"   NLLB target code: {tgt_nllb}")

# Create model
model = EmotionSemanticNMT(
    config,
    model_type='nllb',
    use_emotion=True,
    use_semantic=True,
    use_style=True
).to(device)

# Set tokenizer language codes
model.tokenizer.src_lang = src_nllb
model.tokenizer.tgt_lang = tgt_nllb

print(f"   ✅ Model created with all modules")

# ============================================================================
# CREATE DATASETS
# ============================================================================
print(f"\n2️⃣ Loading annotated dataset for {TRANSLATION_PAIR}...")

try:
    train_dataset = BHT25AnnotatedDataset(
        CSV_PATH,
        model.tokenizer,
        TRANSLATION_PAIR,
        config.MAX_LENGTH,
        'train',
        'nllb'
    )
    val_dataset = BHT25AnnotatedDataset(
        CSV_PATH,
        model.tokenizer,
        TRANSLATION_PAIR,
        config.MAX_LENGTH,
        'val',
        'nllb'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"   ✅ Train samples: {len(train_dataset)}")
    print(f"   ✅ Val samples: {len(val_dataset)}")
    
except Exception as e:
    print(f"❌ ERROR creating datasets: {e}")
    raise

# ============================================================================
# COMPATIBLE SAVE FUNCTION
# ============================================================================
def save_checkpoint_compatible(filepath, data):
    """Save checkpoint compatible with PyTorch version"""
    try:
        # Try with weights_only for PyTorch >= 2.6
        torch.save(data, filepath, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch
        torch.save(data, filepath)
    except Exception as e:
        # Another fallback - save only state dict
        print(f"⚠️ Standard save failed: {e}")
        print("   Saving only model state dict...")
        torch.save(data['model_state_dict'], filepath.replace('.pt', '_state_dict.pt'))

# ============================================================================
# TRAINING LOOP
# ============================================================================
print(f"\n3️⃣ Starting training for {TRANSLATION_PAIR}...")

# Create directories
os.makedirs('./checkpoints', exist_ok=True)
os.makedirs('./outputs', exist_ok=True)

# Create trainer
trainer = Trainer(model, config, TRANSLATION_PAIR)

# Training history
training_history = []

for epoch in range(config.EPOCHS['phase1']):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{config.EPOCHS['phase1']}")
    print(f"{'='*70}")

    # Train
    train_loss = trainer.train_epoch(train_loader, epoch)
    training_history.append({'epoch': epoch+1, 'train_loss': train_loss})
    
    print(f"\n✅ Epoch {epoch+1} completed - Train Loss: {train_loss:.4f}")

    # Validation
    print(f"\n📊 Running validation...")
    evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, TRANSLATION_PAIR)
    metrics, _, _, _ = evaluator.evaluate(val_loader)

    print(f"\nValidation Results:")
    print(f"   BLEU:    {metrics.get('bleu', 0):.2f}")
    print(f"   METEOR:  {metrics.get('meteor', 0):.2f}")
    print(f"   chrF:    {metrics.get('chrf', 0):.2f}")
    if config.ALPHA > 0:
        print(f"   Emotion: {metrics.get('emotion_accuracy', 0):.2f}%")
    if config.BETA > 0:
        print(f"   Semantic: {metrics.get('semantic_score', 0):.4f}")

    # Save checkpoint
    checkpoint_path = f"./checkpoints/esa_nmt_{TRANSLATION_PAIR}_epoch{epoch+1}.pt"
    checkpoint_data = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'training_history': training_history,
        'translation_pair': TRANSLATION_PAIR,
        'nllb_src': src_nllb,
        'nllb_tgt': tgt_nllb
    }
    
    save_checkpoint_compatible(checkpoint_path, checkpoint_data)
    print(f"   💾 Checkpoint saved: {checkpoint_path}")

    # Memory cleanup
    torch.cuda.empty_cache()
    gc.collect()

print(f"\n{'='*70}")
print(f"✅ TRAINING COMPLETED!")
print(f"{'='*70}")

# ============================================================================
# FINAL EVALUATION
# ============================================================================
print(f"\n4️⃣ Final evaluation on validation set...")
evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, TRANSLATION_PAIR)
final_metrics, predictions, references, sources = evaluator.evaluate(val_loader)

print(f"\n📊 FINAL RESULTS:")
print(f"{'='*70}")
print(f"BLEU:     {final_metrics.get('bleu', 0):.2f}")
print(f"METEOR:   {final_metrics.get('meteor', 0):.2f}")
print(f"chrF:     {final_metrics.get('chrf', 0):.2f}")
print(f"ROUGE-L:  {final_metrics.get('rouge_l', 0):.2f}")
if config.ALPHA > 0:
    print(f"Emotion:  {final_metrics.get('emotion_accuracy', 0):.2f}%")
if config.BETA > 0:
    print(f"Semantic: {final_metrics.get('semantic_score', 0):.4f}")
print(f"{'='*70}")

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================
final_path = f"./checkpoints/final_esa_nmt_{TRANSLATION_PAIR}.pt"
final_data = {
    'model_state_dict': model.state_dict(),
    'config': config,
    'metrics': final_metrics,
    'training_history': training_history,
    'translation_pair': TRANSLATION_PAIR,
    'nllb_src': src_nllb,
    'nllb_tgt': tgt_nllb,
    'timestamp': datetime.now().isoformat()
}

save_checkpoint_compatible(final_path, final_data)
print(f"\n💾 Final model saved: {final_path}")

# Save results
results_path = f"./outputs/training_results_{TRANSLATION_PAIR}.json"
with open(results_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)

print(f"📄 Results saved: {results_path}")

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  ✅ TRAINING COMPLETE!                                        ║
║  Language pair: {TRANSLATION_PAIR:<20}                     ║
╚═══════════════════════════════════════════════════════════════╝
""")
