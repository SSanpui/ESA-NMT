"""
Enhanced Ablation Study for ESA-NMT
Fixed for PyTorch compatibility
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
from src.emotion_semantic_nmt_enhanced import Config

# Allow the Config class to be loaded safely
try:
    torch.serialization.add_safe_globals([Config])
except:
    pass  # Older PyTorch version

from src.emotion_semantic_nmt_enhanced import config, EmotionSemanticNMT, ComprehensiveEvaluator
from src.dataset_with_annotations import BHT25AnnotatedDataset
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
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================
TRANSLATION_PAIR = 'hi-te'  # Change as needed
print(f"\nTranslation pair: {TRANSLATION_PAIR}")
print(f"Device: {device}")

# =============================================================================
# COMPATIBLE LOAD FUNCTION
# =============================================================================
def load_checkpoint_compatible(filepath, device):
    """Load checkpoint compatible with PyTorch version"""
    try:
        # Try with weights_only for PyTorch >= 2.6
        return torch.load(filepath, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch
        return torch.load(filepath, map_location=device)
    except Exception as e:
        print(f"⚠️ Load failed: {e}")
        return None

# =============================================================================
# FIND CHECKPOINTS
# =============================================================================
def find_checkpoint(lang_pair):
    """Find checkpoint for language pair"""
    import glob
    
    patterns = [
        f'./checkpoints/final_esa_nmt_{lang_pair}.pt',
        f'./checkpoints/esa_nmt_{lang_pair}_epoch*.pt',
        f'./checkpoints/latest_{lang_pair}.pt'
    ]
    
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            return matches[0]
    
    return None

full_model_checkpoint = find_checkpoint(TRANSLATION_PAIR)

if full_model_checkpoint:
    print(f"✅ Found checkpoint: {full_model_checkpoint}")
else:
    print(f"⚠️ No checkpoint found for {TRANSLATION_PAIR}")
    print("   Will use pre-trained NLLB weights")

# =============================================================================
# ABLATION CONFIGURATIONS
# =============================================================================
configs = [
    {
        'name': 'Base NLLB (Baseline)',
        'emotion': False,
        'semantic': False,
        'checkpoint': None,
        'description': 'No emotion or semantic modules'
    },
    {
        'name': 'Base + Emotion',
        'emotion': True,
        'semantic': False,
        'checkpoint': None,
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
        'checkpoint': full_model_checkpoint,
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
    print(f"{'='*80}")
    
    try:
        # Create model
        print(f"Creating model...")
        
        model = EmotionSemanticNMT(
            config,
            model_type='nllb',
            use_emotion=config_info['emotion'],
            use_semantic=config_info['semantic'],
            use_style=False
        ).to(device)
        
        # Set language codes
        def get_nllb_lang_code(lang):
            lang_map = {'bn': 'ben_Beng', 'hi': 'hin_Deva', 'te': 'tel_Telu'}
            return lang_map.get(lang, f'{lang}_Latn')
        
        src_lang, tgt_lang = translation_pair.split('-')
        src_nllb = get_nllb_lang_code(src_lang)
        tgt_nllb = get_nllb_lang_code(tgt_lang)
        
        model.tokenizer.src_lang = src_nllb
        model.tokenizer.tgt_lang = tgt_nllb
        
        # Load checkpoint if provided
        if config_info['checkpoint'] and os.path.exists(config_info['checkpoint']):
            print(f"📥 Loading checkpoint...")
            checkpoint = load_checkpoint_compatible(config_info['checkpoint'], device)
            if checkpoint and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✅ Checkpoint loaded")
            else:
                print("⚠️ Could not load checkpoint, using pre-trained")
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
        metrics, _, _, _ = evaluator.evaluate(test_loader)
        
        # Print results
        print(f"\n✅ Results:")
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
print(f"\n🔬 Starting ablation study...")

results = {}
for i, config_info in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}]")
    metrics = evaluate_config(config_info, TRANSLATION_PAIR)
    if metrics:
        results[config_info['name']] = metrics

# =============================================================================
# DISPLAY RESULTS
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

print(f"💾 Results saved: {output_file}")

print("\n" + "="*80)
print("✅ ABLATION STUDY COMPLETE!")
print("="*80)
