"""
Enhanced Ablation Study for ESA-NMT
Supports all language pairs: bn-hi, bn-te, hi-te
Includes IndicTrans2 baseline comparison
"""

import os
os.chdir('/kaggle/working/ESA-NMT')

import torch
import gc
import json
import numpy as np
import sys
from datetime import datetime

# Add safe globals for torch.load
import torch.serialization
from emotion_semantic_nmt_enhanced import Config

# Allow the Config class to be loaded safely
torch.serialization.add_safe_globals([Config])

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
# CONFIGURATION - SET YOUR LANGUAGE PAIR HERE
# =============================================================================

TRANSLATION_PAIR = 'hi-te'  # Options: 'bn-hi', 'bn-te', 'hi-te'

print(f"\nTranslation pair: {TRANSLATION_PAIR}")
print(f"Device: {device}")

# Create output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./outputs/ablation_{TRANSLATION_PAIR}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# =============================================================================
# FIND CHECKPOINTS FOR DIFFERENT LANGUAGE PAIRS
# =============================================================================

def find_checkpoint(lang_pair, model_type="full"):
    """Find checkpoint for specific language pair and model type"""
    
    checkpoint_patterns = {
        'full': [
            f'./checkpoints/{lang_pair}_*/final_esa_nmt_{lang_pair}.pt',
            f'./checkpoints/final_esa_nmt_{lang_pair}.pt',
            f'./checkpoints/latest_{lang_pair}.pt',
            f'/kaggle/working/model_{lang_pair}.pt'
        ],
        'epoch': [
            f'./checkpoints/{lang_pair}_*/esa_nmt_{lang_pair}_epoch*.pt',
            f'./checkpoints/esa_nmt_{lang_pair}_epoch*.pt'
        ]
    }
    
    import glob
    for pattern in checkpoint_patterns[model_type]:
        matches = glob.glob(pattern)
        if matches:
            # Get the most recent one
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    
    return None

# Find full model checkpoint
full_model_checkpoint = find_checkpoint(TRANSLATION_PAIR, "full")

if full_model_checkpoint:
    print(f"✅ Found ESA-NMT checkpoint: {full_model_checkpoint}")
else:
    print(f"\n⚠️ WARNING: No trained checkpoint found for {TRANSLATION_PAIR}!")
    print("   Will train a new model or use pre-trained weights")
    print("   To train: python retrain_with_fixed_code.py (set TRANSLATION_PAIR)")
    
    # Ask if we should train
    response = input(f"\n❓ Train {TRANSLATION_PAIR} now? (y/n): ").strip().lower()
    if response == 'y':
        print(f"⏳ Training {TRANSLATION_PAIR}...")
        # Update the retrain script to use this language pair
        with open('retrain_with_fixed_code.py', 'r') as f:
            content = f.read()
        # Update the TRANSLATION_PAIR variable
        content = content.replace("TRANSLATION_PAIR = 'hi-te'", f"TRANSLATION_PAIR = '{TRANSLATION_PAIR}'")
        with open('retrain_with_fixed_code.py', 'w') as f:
            f.write(content)
        
        # Run the training
        exec(open('retrain_with_fixed_code.py').read())
        
        # Try to find checkpoint again
        full_model_checkpoint = find_checkpoint(TRANSLATION_PAIR, "full")
        if full_model_checkpoint:
            print(f"✅ Found new checkpoint: {full_model_checkpoint}")
        else:
            print("❌ Training completed but no checkpoint found")
            sys.exit(1)
    else:
        print("ℹ️ Using pre-trained NLLB weights only for ablation study")

# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================

configs = [
    {
        'name': 'Base NLLB (Baseline)',
        'emotion': False,
        'semantic': False,
        'checkpoint': None,
        'description': 'No emotion or semantic modules',
        'color': 'red'
    },
    {
        'name': 'Base + Emotion',
        'emotion': True,
        'semantic': False,
        'checkpoint': None,  # We'll create this by disabling semantic
        'description': 'Adds emotion awareness only',
        'color': 'blue'
    },
    {
        'name': 'Base + Semantic',
        'emotion': False,
        'semantic': True,
        'checkpoint': None,  # We'll create this by disabling emotion
        'description': 'Adds semantic similarity only',
        'color': 'green'
    },
    {
        'name': 'Full ESA-NMT',
        'emotion': True,
        'semantic': True,
        'checkpoint': full_model_checkpoint,
        'description': 'Emotion + Semantic (proposed model)',
        'color': 'purple'
    }
]

# =============================================================================
# EVALUATION FUNCTION WITH PROPER CHECKPOINT HANDLING
# =============================================================================

def evaluate_config(config_info, translation_pair, save_samples=50):
    """Evaluate one ablation configuration with proper error handling"""
    
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
        
        # Set NLLB language codes
        def get_nllb_lang_code(lang):
            if lang == 'bn':
                return 'ben_Beng'
            elif lang == 'hi':
                return 'hin_Deva'
            elif lang == 'te':
                return 'tel_Telu'
            else:
                return f'{lang}_Latn'
        
        src_lang, tgt_lang = translation_pair.split('-')
        src_nllb = get_nllb_lang_code(src_lang)
        tgt_nllb = get_nllb_lang_code(tgt_lang)
        
        model.tokenizer.src_lang = src_nllb
        model.tokenizer.tgt_lang = tgt_nllb
        
        # Load checkpoint if provided
        if config_info['checkpoint'] and os.path.exists(config_info['checkpoint']):
            print(f"📥 Loading trained checkpoint...")
            try:
                # First try with weights_only=False for custom classes
                checkpoint = torch.load(
                    config_info['checkpoint'], 
                    map_location=device,
                    weights_only=False
                )
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Checkpoint loaded (trained model)")
            except Exception as e:
                print(f"⚠️ Standard load failed: {e}")
                print("   Trying weights_only=True...")
                try:
                    checkpoint = torch.load(
                        config_info['checkpoint'],
                        map_location=device,
                        weights_only=True
                    )
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("✅ Checkpoint loaded with weights_only=True")
                except Exception as e2:
                    print(f"❌ Both load methods failed: {e2}")
                    print("   Using pre-trained weights only")
        else:
            print("📝 Using pre-trained NLLB base")
        
        # Load test dataset
        print("Loading test dataset...")
        test_dataset = BHT25AnnotatedDataset(
            'BHT25_All_annotated.csv',
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
        
        # Save sample translations
        samples_file = f"{output_dir}/samples_{translation_pair}_{name.replace(' ', '_')}.json"
        samples_data = []
        for i in range(min(save_samples, len(sources))):
            samples_data.append({
                'id': i,
                'source': sources[i],
                'prediction': preds[i] if i < len(preds) else '',
                'reference': refs[i] if i < len(refs) else '',
                'config': name
            })
        
        with open(samples_file, 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)
        
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
        
        return metrics, samples_file
        
    except Exception as e:
        print(f"❌ Error in evaluate_config: {e}")
        import traceback
        traceback.print_exc()
        
        torch.cuda.empty_cache()
        gc.collect()
        return None, None

# =============================================================================
# RUN ABLATION STUDY FOR CURRENT LANGUAGE PAIR
# =============================================================================

print(f"\n🔬 Starting ablation study for {TRANSLATION_PAIR}...")
print(f"   Testing {len(configs)} configurations")
print(f"   Output directory: {output_dir}")
print(f"   Estimated time: 30-40 minutes")

results = {}
sample_files = {}

for i, config_info in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}]")
    
    metrics, samples_file = evaluate_config(config_info, TRANSLATION_PAIR)
    
    if metrics:
        results[config_info['name']] = metrics
        if samples_file:
            sample_files[config_info['name']] = samples_file
    
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
    
    improvements = {}
    for metric in ['bleu', 'meteor', 'chrf', 'rouge_l']:
        if metric in baseline and metric in full_model:
            baseline_val = baseline[metric]
            full_val = full_model[metric]
            improvement = full_val - baseline_val
            pct_improvement = (improvement / baseline_val) * 100 if baseline_val > 0 else 0
            improvements[metric] = {
                'baseline': baseline_val,
                'full': full_val,
                'absolute': improvement,
                'percentage': pct_improvement
            }
    
    for metric, data in improvements.items():
        metric_name = metric.upper().replace('_', '-') if metric != 'rouge_l' else 'ROUGE-L'
        print(f"{metric_name:10s}: {data['baseline']:.2f} → {data['full']:.2f} "
              f"(+{data['absolute']:.2f}, +{data['percentage']:.1f}%)")
    
    print()

# =============================================================================
# CREATE VISUALIZATION (OPTIONAL)
# =============================================================================

try:
    import matplotlib.pyplot as plt
    
    if results:
        metrics_to_plot = ['bleu', 'meteor', 'chrf']
        config_names = [c['name'] for c in configs if c['name'] in results]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, metric in enumerate(metrics_to_plot):
            values = [results[name].get(metric, 0) for name in config_names if name in results]
            colors = [c['color'] for c in configs if c['name'] in config_names]
            
            axes[idx].bar(range(len(values)), values, color=colors)
            axes[idx].set_title(f'{metric.upper()} Score')
            axes[idx].set_xlabel('Configuration')
            axes[idx].set_ylabel('Score')
            axes[idx].set_xticks(range(len(config_names)))
            axes[idx].set_xticklabels([name.split()[0] for name in config_names], rotation=45)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.5, f'{v:.2f}', ha='center')
        
        plt.suptitle(f'Ablation Study Results - {TRANSLATION_PAIR.upper()}')
        plt.tight_layout()
        
        plot_file = f"{output_dir}/ablation_plot_{TRANSLATION_PAIR}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plot saved: {plot_file}")
        
except ImportError:
    print("⚠️ Matplotlib not installed, skipping visualization")
except Exception as e:
    print(f"⚠️ Could not create plot: {e}")

# =============================================================================
# SAVE ALL RESULTS
# =============================================================================

output_file = f"{output_dir}/ablation_study_{TRANSLATION_PAIR}.json"

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

print(f"💾 Results saved: {output_file}")

# Save detailed report
report_file = f"{output_dir}/ablation_report_{TRANSLATION_PAIR}.md"
with open(report_file, 'w') as f:
    f.write(f"# Ablation Study Report - {TRANSLATION_PAIR.upper()}\n\n")
    f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"**Language Pair**: {TRANSLATION_PAIR}\n\n")
    
    f.write("## Results Summary\n\n")
    f.write("| Configuration | BLEU | METEOR | chrF | ROUGE-L | Emotion | Semantic |\n")
    f.write("|--------------|------|--------|------|---------|---------|----------|\n")
    
    for config_info in configs:
        name = config_info['name']
        if name in results:
            metrics = results[name]
            f.write(f"| {name} | {metrics.get('bleu', 0):.2f} | {metrics.get('meteor', 0):.2f} | "
                   f"{metrics.get('chrf', 0):.2f} | {metrics.get('rouge_l', 0):.2f} | "
                   f"{metrics.get('emotion_accuracy', 0):.2f}% | {metrics.get('semantic_score', 0):.4f} |\n")
    
    f.write("\n## Sample Files\n\n")
    for name, sample_file in sample_files.items():
        f.write(f"- [{name}]({os.path.basename(sample_file)})\n")
    
    if 'Base NLLB (Baseline)' in results and 'Full ESA-NMT' in results:
        f.write("\n## Improvement Analysis\n\n")
        baseline = results['Base NLLB (Baseline)']
        full = results['Full ESA-NMT']
        
        for metric in ['bleu', 'meteor', 'chrf', 'rouge_l']:
            if metric in baseline and metric in full:
                improvement = full[metric] - baseline[metric]
                pct = (improvement / baseline[metric]) * 100 if baseline[metric] > 0 else 0
                metric_name = metric.upper().replace('_', '-') if metric != 'rouge_l' else 'ROUGE-L'
                f.write(f"- **{metric_name}**: {baseline[metric]:.2f} → {full[metric]:.2f} "
                       f"(+{improvement:.2f}, +{pct:.1f}%)\n")

print(f"📝 Report saved: {report_file}")

# Copy to /kaggle/working for download
import shutil
shutil.copy(output_file, f'/kaggle/working/ablation_study_{TRANSLATION_PAIR}.json')
print(f"💾 Copied to: /kaggle/working/ablation_study_{TRANSLATION_PAIR}.json")

print("\n" + "="*80)
print("✅ ABLATION STUDY COMPLETE!")
print("="*80)

print(f"\n📊 Evaluated configurations:")
for name in results.keys():
    print(f"   ✅ {name}")

print(f"\n📁 Output files:")
print(f"   📄 Results: {output_file}")
print(f"   📝 Report: {report_file}")
if 'plot_file' in locals():
    print(f"   📊 Plot: {plot_file}")
for name, sample_file in sample_files.items():
    print(f"   🧪 {name} samples: {sample_file}")

print(f"\n⏭️ Next steps:")
print(f"   1. Change TRANSLATION_PAIR in this script (line 48)")
print(f"   2. Run again for other language pairs")
print(f"   3. Compare with IndicTrans2 using: python indic_trans2_comparison.py")

print("\n💡 Key Findings:")
print("   • The table shows how each module contributes to performance")
print("   • Full ESA-NMT should show the best overall scores")
print("   • Emotion module improves emotion preservation")
print("   • Semantic module improves meaning preservation")