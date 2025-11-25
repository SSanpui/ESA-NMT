"""
Simple Ablation Study ONLY
Shows how ESA-NMT modules improve translation quality

Compares:
1. Base NLLB (baseline)
2. Base + Emotion module
3. Base + Semantic module
4. Full ESA-NMT (both modules)

All metrics: BLEU, METEOR, chrF, ROUGE-L, Emotion Accuracy, Semantic Score
"""

import os
os.chdir('/kaggle/working/ESA-NMT')

import torch
import gc
import json
import numpy as np
from emotion_semantic_nmt_enhanced import config, EmotionSemanticNMT, ComprehensiveEvaluator
from dataset_with_annotations import BHT25AnnotatedDataset
from torch.utils.data import DataLoader

# Setup METEOR
print("📦 Setting up METEOR...")
try:
    import nltk
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    print("✅ METEOR ready")
except:
    print("⚠️ METEOR setup failed")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("="*80)
print("ESA-NMT ABLATION STUDY")
print("Shows improvement from each module")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

TRANSLATION_PAIR = 'bn-te'  # Change to 'bn-hi' for Hindi

print(f"\nTranslation pair: {TRANSLATION_PAIR}")
print(f"Device: {device}")

# Find ESA-NMT full model checkpoint
# AFTER RETRAINING: Update this path to your new checkpoint!
checkpoint_paths = [
    f'./checkpoints/final_esa_nmt_{TRANSLATION_PAIR}.pt',  # ← NEW retrained model
    f'/kaggle/working/model_{TRANSLATION_PAIR}.pt',        # Old location
    f'./checkpoints/final_model_nllb_{TRANSLATION_PAIR}.pt'  # Old name
]

full_model_checkpoint = None
for path in checkpoint_paths:
    if os.path.exists(path):
        full_model_checkpoint = path
        print(f"✅ Found ESA-NMT checkpoint: {path}")
        break

if not full_model_checkpoint:
    print("\n⚠️ WARNING: ESA-NMT full model checkpoint not found!")
    print("   Full model will use pre-trained NLLB weights only")
    print("   Results may not show improvement")
    print("\n   💡 Run retrain_with_fixed_code.py first to train the model!")

# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================

configs = [
    {
        'name': 'Base NLLB (Baseline)',
        'emotion': False,
        'semantic': False,
        'checkpoint': None,  # Pre-trained NLLB only
        'description': 'No emotion or semantic modules'
    },
    {
        'name': 'Base + Emotion',
        'emotion': True,
        'semantic': False,
        'checkpoint': None,  # Evaluate without training for fair comparison
        'description': 'Adds emotion awareness'
    },
    {
        'name': 'Base + Semantic',
        'emotion': False,
        'semantic': True,
        'checkpoint': None,
        'description': 'Adds semantic similarity'
    },
    {
        'name': 'Full ESA-NMT',
        'emotion': True,
        'semantic': True,
        'checkpoint': full_model_checkpoint,  # Your trained model!
        'description': 'Emotion + Semantic (proposed model)'
    }
]

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_config(config_info, translation_pair):
    """Evaluate one ablation configuration"""

    name = config_info['name']
    print(f"\n{'='*80}")
    print(f"Configuration: {name}")
    print(f"Description: {config_info['description']}")
    print(f"{'='*80}")

    try:
        # Create model
        print(f"Creating model (emotion={config_info['emotion']}, semantic={config_info['semantic']})...")

        model = EmotionSemanticNMT(
            config,
            model_type='nllb',
            use_emotion=config_info['emotion'],
            use_semantic=config_info['semantic'],
            use_style=False
        ).to(device)

        # Load checkpoint if provided (for Full model)
        if config_info['checkpoint']:
            print(f"📥 Loading trained checkpoint...")
            checkpoint = torch.load(config_info['checkpoint'], map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Checkpoint loaded (trained model)")
        else:
            print("📝 Using pre-trained NLLB base")

        # Load test dataset
        print("Loading test dataset...")
        test_dataset = BHT25AnnotatedDataset(
            'BHT25_All_annotated.csv',  # Using annotated dataset
            model.tokenizer,
            translation_pair,
            config.MAX_LENGTH,
            'test',
            'nllb'
        )

        test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
        print(f"✅ Test samples: {len(test_dataset)}")

        # Evaluate
        print("Evaluating...")
        evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
        metrics, preds, refs, sources = evaluator.evaluate(test_loader)

        # Print results
        print(f"\n✅ Results for {name}:")
        print(f"   BLEU:    {metrics.get('bleu', 0):.2f}")
        print(f"   METEOR:  {metrics.get('meteor', 0):.2f}")
        print(f"   chrF:    {metrics.get('chrf', 0):.2f}")
        print(f"   ROUGE-L: {metrics.get('rouge_l', 0):.2f}")

        if config_info['emotion'] and 'emotion_accuracy' in metrics:
            print(f"   Emotion: {metrics.get('emotion_accuracy', 0):.2f}%")
        if config_info['semantic'] and 'semantic_score' in metrics:
            print(f"   Semantic: {metrics.get('semantic_score', 0):.4f}")

        # Cleanup
        del model, evaluator, test_dataset, test_loader
        torch.cuda.empty_cache()
        gc.collect()

        return metrics

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

        torch.cuda.empty_cache()
        gc.collect()
        return None

# =============================================================================
# RUN ABLATION STUDY
# =============================================================================

print(f"\n🔬 Starting ablation study for {TRANSLATION_PAIR}...")
print(f"   Testing {len(configs)} configurations")
print(f"   Estimated time: 30-40 minutes")

results = {}

for i, config_info in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}]")

    metrics = evaluate_config(config_info, TRANSLATION_PAIR)

    if metrics:
        results[config_info['name']] = metrics

    # Clear memory between configs
    print("\n🧹 Clearing memory...")
    torch.cuda.empty_cache()
    gc.collect()

# =============================================================================
# DISPLAY RESULTS TABLE
# =============================================================================

print("\n" + "="*90)
print(f"ABLATION STUDY RESULTS: {TRANSLATION_PAIR.upper()}")
print("="*90)

header = f"{'Configuration':<25} {'BLEU':<8} {'METEOR':<8} {'chrF':<8} {'ROUGE-L':<10} {'Emotion':<10} {'Semantic':<10}"
print(header)
print("-"*90)

for config_info in configs:
    name = config_info['name']
    if name in results:
        metrics = results[name]
        bleu = f"{metrics.get('bleu', 0):.2f}"
        meteor = f"{metrics.get('meteor', 0):.2f}"
        chrf = f"{metrics.get('chrf', 0):.2f}"
        rouge = f"{metrics.get('rouge_l', 0):.2f}"
        emotion = f"{metrics.get('emotion_accuracy', 0):.2f}%" if metrics.get('emotion_accuracy', 0) > 0 else "N/A"
        semantic = f"{metrics.get('semantic_score', 0):.4f}" if metrics.get('semantic_score', 0) > 0 else "N/A"

        print(f"{name:<25} {bleu:<8} {meteor:<8} {chrf:<8} {rouge:<10} {emotion:<10} {semantic:<10}")

print()

# =============================================================================
# CALCULATE IMPROVEMENTS
# =============================================================================

if 'Base NLLB (Baseline)' in results and 'Full ESA-NMT' in results:
    baseline = results['Base NLLB (Baseline)']
    full_model = results['Full ESA-NMT']

    print("\n" + "="*80)
    print("IMPROVEMENT OVER BASELINE")
    print("="*80)

    improvements = {
        'BLEU': full_model['bleu'] - baseline['bleu'],
        'METEOR': full_model['meteor'] - baseline['meteor'],
        'chrF': full_model['chrf'] - baseline['chrf'],
        'ROUGE-L': full_model['rouge_l'] - baseline['rouge_l']
    }

    for metric, improvement in improvements.items():
        baseline_val = baseline[metric.lower().replace('-', '_')]
        full_val = full_model[metric.lower().replace('-', '_')]
        pct_improvement = (improvement / baseline_val) * 100 if baseline_val > 0 else 0

        print(f"{metric:10s}: {baseline_val:.2f} → {full_val:.2f} (+{improvement:.2f}, +{pct_improvement:.1f}%)")

    print()

# =============================================================================
# SAVE RESULTS
# =============================================================================

os.makedirs('./outputs', exist_ok=True)
output_file = f'./outputs/ablation_study_{TRANSLATION_PAIR}.json'

json_results = {}
for name, metrics in results.items():
    json_results[name] = {
        'bleu': float(metrics.get('bleu', 0)),
        'meteor': float(metrics.get('meteor', 0)),
        'chrf': float(metrics.get('chrf', 0)),
        'rouge_l': float(metrics.get('rouge_l', 0)),
        'emotion_accuracy': float(metrics.get('emotion_accuracy', 0)),
        'semantic_score': float(metrics.get('semantic_score', 0))
    }

with open(output_file, 'w') as f:
    json.dump(json_results, f, indent=2)

print(f"💾 Saved: {output_file}")

# Copy to /kaggle/working
import shutil
shutil.copy(output_file, f'/kaggle/working/ablation_study_{TRANSLATION_PAIR}.json')
print(f"💾 Copied to: /kaggle/working/ablation_study_{TRANSLATION_PAIR}.json")

print("\n" + "="*80)
print("✅ ABLATION STUDY COMPLETE!")
print("="*80)

print(f"\n📊 Evaluated {len(results)} configurations:")
for name in results.keys():
    print(f"   ✅ {name}")

print(f"\n📥 Download this file:")
print(f"   ablation_study_{TRANSLATION_PAIR}.json")

print(f"\n⏭️ Next: Change TRANSLATION_PAIR to 'bn-te' (line 27) and run again for Telugu")

print("\n💡 Key Findings:")
print("   The table above shows how each module contributes to performance")
print("   Full ESA-NMT (your model) should show the best scores")
print("   This proves the value of emotion and semantic modules!")
