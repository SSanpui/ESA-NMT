#!/usr/bin/env python3
"""
Retrain ESA-NMT with properly integrated emotion/semantic modules
This fixes the bug where modules were only used for loss, not generation
Now supports all language pairs: bn-hi, bn-te, hi-te
"""

import torch
import gc
import os
import json
from datetime import datetime

# Import from main file
from emotion_semantic_nmt_enhanced import (
    Config, EmotionSemanticNMT, Trainer, ComprehensiveEvaluator,
    device, config
)
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

print("""
╔═══════════════════════════════════════════════════════════════╗
║  RETRAIN ESA-NMT with Fixed Code (GAMMA=0.5)                 ║
║  Supports: bn-hi, bn-te, hi-te                               ║
╚═══════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# CONFIGURATION - CHANGE THIS FOR DIFFERENT LANGUAGE PAIRS
# ============================================================================
TRANSLATION_PAIR = 'hi-te'  # Options: 'bn-hi', 'bn-te', 'hi-te'
CSV_PATH = 'BHT25_All_annotated.csv'  # Your annotated dataset

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
# LOAD DATASET AND CHECK AVAILABILITY
# ============================================================================
print(f"\n🔍 Checking dataset for {TRANSLATION_PAIR}...")

# Check which columns are available in the dataset
import pandas as pd
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Dataset loaded with {len(df)} rows")
    
    # Check for required columns based on language pair
    src_lang, tgt_lang = TRANSLATION_PAIR.split('-')
    
    # Map language codes to column names
    lang_to_col = {
        'bn': 'bn',
        'hi': 'hi', 
        'te': 'te'
    }
    
    src_col = lang_to_col[src_lang]
    tgt_col = lang_to_col[tgt_lang]
    
    # Check if columns exist
    missing_cols = []
    for col in [src_col, tgt_col]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"❌ ERROR: Missing columns for {TRANSLATION_PAIR}: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        exit(1)
    
    # Check for annotation columns
    emotion_src_col = f'emotion_{src_lang}'
    emotion_tgt_col = f'emotion_{tgt_lang}'
    semantic_col = f'semantic_{src_lang}_{tgt_lang}'
    
    print(f"   Source text column: {src_col}")
    print(f"   Target text column: {tgt_col}")
    print(f"   Emotion source column: {emotion_src_col} ({'found' if emotion_src_col in df.columns else 'not found'})")
    print(f"   Emotion target column: {emotion_tgt_col} ({'found' if emotion_tgt_col in df.columns else 'not found'})")
    print(f"   Semantic similarity column: {semantic_col} ({'found' if semantic_col in df.columns else 'not found'})")
    
    # Count non-empty rows
    valid_rows = df.dropna(subset=[src_col, tgt_col])
    print(f"   Valid parallel sentences: {len(valid_rows)}")
    
else:
    print(f"❌ ERROR: Dataset not found at {CSV_PATH}")
    exit(1)

# ============================================================================
# CREATE MODEL WITH ALL MODULES
# ============================================================================
print(f"\n1️⃣ Creating ESA-NMT model for {TRANSLATION_PAIR} with all modules...")
model = EmotionSemanticNMT(
    config,
    model_type='nllb',
    use_emotion=True,   # ← Enabled
    use_semantic=True,  # ← Enabled
    use_style=True      # ← Enabled
).to(device)

print(f"   ✅ Model created with emotion, semantic, and style modules")

# ============================================================================
# CREATE DATASETS
# ============================================================================
print(f"\n2️⃣ Loading annotated dataset for {TRANSLATION_PAIR}...")

# Determine NLLB language codes based on language pair
def get_nllb_lang_code(lang):
    """Convert language code to NLLB format"""
    if lang == 'bn':
        return 'ben_Beng'
    elif lang == 'hi':
        return 'hin_Deva'
    elif lang == 'te':
        return 'tel_Telu'
    else:
        return f'{lang}_Latn'

src_lang, tgt_lang = TRANSLATION_PAIR.split('-')
src_nllb = get_nllb_lang_code(src_lang)
tgt_nllb = get_nllb_lang_code(tgt_lang)

print(f"   NLLB source code: {src_nllb}")
print(f"   NLLB target code: {tgt_nllb}")

# Update tokenizer for correct language direction
model.tokenizer.src_lang = src_nllb
model.tokenizer.tgt_lang = tgt_nllb

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
    print(f"   Make sure your dataset has columns for {TRANSLATION_PAIR}")
    exit(1)

# ============================================================================
# CREATE TRAINER AND TRAIN
# ============================================================================
print(f"\n3️⃣ Starting training for {TRANSLATION_PAIR}...")

# Create output directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./outputs/{TRANSLATION_PAIR}_{timestamp}"
checkpoint_dir = f"./checkpoints/{TRANSLATION_PAIR}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Save configuration
config_info = {
    'translation_pair': TRANSLATION_PAIR,
    'source_language': src_lang,
    'target_language': tgt_lang,
    'nllb_source': src_nllb,
    'nllb_target': tgt_nllb,
    'timestamp': timestamp,
    'train_samples': len(train_dataset),
    'val_samples': len(val_dataset),
    'hyperparameters': {
        'alpha': config.ALPHA,
        'beta': config.BETA,
        'gamma': config.GAMMA,
        'delta': config.DELTA,
        'epochs': config.EPOCHS['phase1'],
        'batch_size': config.BATCH_SIZE,
        'max_length': config.MAX_LENGTH
    }
}

with open(f"{output_dir}/config.json", 'w') as f:
    json.dump(config_info, f, indent=2)

trainer = Trainer(model, config, TRANSLATION_PAIR)

# Train for specified epochs
training_history = []
for epoch in range(config.EPOCHS['phase1']):
    print(f"\n{'='*70}")
    print(f"EPOCH {epoch+1}/{config.EPOCHS['phase1']}")
    print(f"{'='*70}")

    train_loss = trainer.train_epoch(train_loader, epoch)
    training_history.append({'epoch': epoch+1, 'train_loss': train_loss})

    print(f"\n✅ Epoch {epoch+1} completed - Train Loss: {train_loss:.4f}")

    # Validation every epoch
    print(f"\n📊 Running validation...")
    evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, TRANSLATION_PAIR)
    metrics, _, _, _ = evaluator.evaluate(val_loader)

    print(f"\nValidation Results:")
    print(f"   BLEU:    {metrics.get('bleu', 0):.2f}")
    print(f"   METEOR:  {metrics.get('meteor', 0):.2f}")
    print(f"   chrF:    {metrics.get('chrf', 0):.2f}")
    print(f"   Emotion: {metrics.get('emotion_accuracy', 0):.2f}%")
    print(f"   Semantic: {metrics.get('semantic_score', 0):.4f}")

    # Save checkpoint after each epoch
    checkpoint_path = f"{checkpoint_dir}/esa_nmt_{TRANSLATION_PAIR}_epoch{epoch+1}.pt"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics,
        'training_history': training_history,
        'translation_pair': TRANSLATION_PAIR,
        'nllb_src': src_nllb,
        'nllb_tgt': tgt_nllb
    }, checkpoint_path, weights_only=False)  # Changed to False for custom classes
    print(f"   💾 Checkpoint saved: {checkpoint_path}")

    # Save epoch results
    epoch_results = {
        'epoch': epoch + 1,
        'train_loss': float(train_loss),
        'validation_metrics': ComprehensiveEvaluator.convert_to_json_serializable(metrics)
    }
    with open(f"{output_dir}/epoch_{epoch+1}_results.json", 'w') as f:
        json.dump(epoch_results, f, indent=2)

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
print(f"Emotion:  {final_metrics.get('emotion_accuracy', 0):.2f}%")
print(f"Semantic: {final_metrics.get('semantic_score', 0):.4f}")
print(f"{'='*70}")

# ============================================================================
# SAVE FINAL MODEL AND RESULTS
# ============================================================================
# Save final model
final_path = f"{checkpoint_dir}/final_esa_nmt_{TRANSLATION_PAIR}.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config,
    'metrics': final_metrics,
    'training_history': training_history,
    'translation_pair': TRANSLATION_PAIR,
    'nllb_src': src_nllb,
    'nllb_tgt': tgt_nllb,
    'timestamp': timestamp
}, final_path, weights_only=False)

print(f"\n💾 Final model saved: {final_path}")

# Save predictions
predictions_file = f"{output_dir}/predictions.json"
with open(predictions_file, 'w') as f:
    pred_data = []
    for i, (src, pred, ref) in enumerate(zip(sources[:100], predictions[:100], references[:100])):  # Save first 100
        pred_data.append({
            'id': i,
            'source': src,
            'prediction': pred,
            'reference': ref
        })
    json.dump(pred_data, f, indent=2, ensure_ascii=False)

print(f"📄 Predictions saved: {predictions_file}")

# Save all results
results_path = f"{output_dir}/training_results_{TRANSLATION_PAIR}.json"
results_data = {
    'config': config_info,
    'final_metrics': ComprehensiveEvaluator.convert_to_json_serializable(final_metrics),
    'training_history': training_history,
    'model_path': final_path
}

with open(results_path, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"📄 Results saved: {results_path}")

# ============================================================================
# CREATE COMPARISON WITH INDICTRANS2
# ============================================================================
print(f"\n5️⃣ Setting up IndicTrans2 comparison...")

# Create a simple script for IndicTrans2 comparison
indic_trans2_script = f"""#!/usr/bin/env python3
'''
IndicTrans2 comparison for {TRANSLATION_PAIR}
Run this after training to compare with ESA-NMT
'''

import sys
sys.path.append('.')

from indic_trans2_comparison import compare_with_indic_trans2

if __name__ == '__main__':
    print(f"Comparing ESA-NMT with IndicTrans2 for {TRANSLATION_PAIR}...")
    
    # Load your trained model
    import torch
    from emotion_semantic_nmt_enhanced import EmotionSemanticNMT, config
    
    model = EmotionSemanticNMT(
        config,
        model_type='nllb',
        use_emotion=True,
        use_semantic=True,
        use_style=True
    ).to('cuda')
    
    checkpoint = torch.load('{final_path}', map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run comparison
    results = compare_with_indic_trans2(
        model=model,
        translation_pair='{TRANSLATION_PAIR}',
        csv_path='{CSV_PATH}',
        num_samples=1000  # Compare on 1000 samples
    )
    
    print(f"\\n✅ Comparison complete!")
    print(f"Results saved to: ./outputs/indic_trans2_comparison_{{TRANSLATION_PAIR}}.json")
"""

with open(f"{output_dir}/run_indic_trans2_comparison.py", 'w') as f:
    f.write(indic_trans2_script)

print(f"💡 Created comparison script: {output_dir}/run_indic_trans2_comparison.py")

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║  ✅ TRAINING COMPLETE!                                        ║
║  Language pair: {TRANSLATION_PAIR:<20}                     ║
║                                                               ║
║  Next steps:                                                 ║
║  1. Run ablation study:                                      ║
║     python ablation_study_only.py                            ║
║                                                               ║
║  2. Compare with IndicTrans2:                                ║
║     python {output_dir}/run_indic_trans2_comparison.py
║                                                               ║
║  3. Train other language pairs:                              ║
║     • Change TRANSLATION_PAIR to 'bn-hi' or 'bn-te'          ║
║     • Run this script again                                  ║
╚═══════════════════════════════════════════════════════════════╝
""")

# Save symlink to latest model
latest_link = f"./checkpoints/latest_{TRANSLATION_PAIR}.pt"
if os.path.exists(latest_link):
    os.remove(latest_link)
os.symlink(final_path, latest_link)
print(f"🔗 Latest model symlink: {latest_link}")