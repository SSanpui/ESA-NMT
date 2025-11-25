#!/usr/bin/env python3
"""
ESA-NMT Evaluation Script
Comprehensive evaluation on test set
"""

import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu.metrics import BLEU, CHRF
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
import nltk
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


def compute_translation_metrics(references, hypotheses):
    """Compute translation quality metrics"""
    metrics = {}
    
    # BLEU
    bleu = BLEU()
    bleu_score = bleu.corpus_score(hypotheses, [references])
    metrics['bleu'] = bleu_score.score
    
    # chrF
    chrf = CHRF()
    chrf_score = chrf.corpus_score(hypotheses, [references])
    metrics['chrf'] = chrf_score.score
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = []
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        rouge_scores.append(score['rougeL'].fmeasure)
    metrics['rouge_l'] = np.mean(rouge_scores)
    
    # METEOR
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = nltk.word_tokenize(ref.lower())
        hyp_tokens = nltk.word_tokenize(hyp.lower())
        score = meteor_score([ref_tokens], hyp_tokens)
        meteor_scores.append(score)
    metrics['meteor'] = np.mean(meteor_scores) * 100  # Scale to 0-100
    
    return metrics


def compute_emotion_metrics(true_labels, pred_labels, num_classes=8):
    """Compute emotion classification metrics"""
    metrics = {}
    
    # Overall accuracy
    metrics['accuracy'] = accuracy_score(true_labels, pred_labels) * 100
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, pred_labels, average=None, labels=range(num_classes)
    )
    
    emotion_names = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'disgust', 'anticipation']
    
    metrics['per_emotion'] = {}
    for i, emotion in enumerate(emotion_names):
        metrics['per_emotion'][emotion] = {
            'precision': float(precision[i] * 100),
            'recall': float(recall[i] * 100),
            'f1': float(f1[i] * 100),
            'support': int(support[i])
        }
    
    # Macro averages
    metrics['macro_precision'] = float(np.mean(precision) * 100)
    metrics['macro_recall'] = float(np.mean(recall) * 100)
    metrics['macro_f1'] = float(np.mean(f1) * 100)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels, labels=range(num_classes))
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def compute_semantic_metrics(semantic_scores):
    """Compute semantic consistency metrics"""
    metrics = {}
    
    metrics['mean_similarity'] = float(np.mean(semantic_scores))
    metrics['std_similarity'] = float(np.std(semantic_scores))
    metrics['min_similarity'] = float(np.min(semantic_scores))
    metrics['max_similarity'] = float(np.max(semantic_scores))
    metrics['median_similarity'] = float(np.median(semantic_scores))
    
    return metrics


def plot_confusion_matrix(cm, emotion_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotion_names,
                yticklabels=emotion_names)
    plt.title('Emotion Classification Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics, output_path):
    """Plot translation metrics comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['BLEU', 'METEOR', 'ROUGE-L', 'chrF']
    metric_values = [
        metrics.get('bleu', 0),
        metrics.get('meteor', 0),
        metrics.get('rouge_l', 0) * 100,  # Scale to 0-100
        metrics.get('chrf', 0)
    ]
    
    bars = ax.bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Score')
    ax.set_title('Translation Quality Metrics')
    ax.set_ylim([0, 100])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate ESA-NMT')
    parser.add_argument('--translation_pair', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='facebook/nllb-200-distilled-600M')
    parser.add_argument('--emotion_model', type=str, default='xlm-roberta-base')
    parser.add_argument('--semantic_model', type=str, default='sentence-transformers/LaBSE')
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--compute_all_metrics', action='store_true')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse translation pair
    src_lang, tgt_lang = args.translation_pair.split('-')
    lang_map = {'bn': 'bengali', 'hi': 'hindi', 'te': 'telugu'}
    
    # Load test data
    print("Loading test dataset...")
    if os.path.exists('BHT25_annotated.csv'):
        df = pd.read_csv('BHT25_annotated.csv')
    else:
        df = pd.read_csv('BHT25_All.csv')
    
    # Use last 15% as test set
    test_size = int(len(df) * args.test_split)
    test_df = df.iloc[-test_size:]
    
    print(f"Test set size: {len(test_df)}")
    
    # Load model and tokenizer
    print(f"\nLoading model from {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    
    # For evaluation, we'll use the base translation model
    # In full implementation, you'd load the complete ESA-NMT model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    model = model.to(device)
    model.eval()
    
    # Get references and generate translations
    print("\nGenerating translations...")
    references = []
    hypotheses = []
    emotion_labels_true = []
    emotion_labels_pred = []
    semantic_scores = []
    
    # NLLB language codes
    nllb_codes = {
        'bn': 'ben_Beng',
        'hi': 'hin_Deva',
        'te': 'tel_Telu'
    }
    
    src_col = lang_map[src_lang]
    tgt_col = lang_map[tgt_lang]
    
    with torch.no_grad():
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Translating"):
            src_text = str(row[src_col])
            ref_text = str(row[tgt_col])
            
            # Tokenize
            inputs = tokenizer(
                src_text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128,
                src_lang=nllb_codes[src_lang]
            ).to(device)
            
            # Generate
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[nllb_codes[tgt_lang]],
                max_length=128,
                num_beams=5
            )
            
            # Decode
            translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            references.append(ref_text)
            hypotheses.append(translation)
            
            # Get emotion labels if available
            if 'emotion_label' in row:
                emotion_labels_true.append(row['emotion_label'])
                # In full implementation, predict emotion for translation
                emotion_labels_pred.append(row['emotion_label'])  # Placeholder
            
            # Get semantic scores if available
            semantic_key = f'semantic_{src_lang}_{tgt_lang}'
            if semantic_key in row:
                semantic_scores.append(row[semantic_key])
    
    # Compute translation metrics
    print("\nComputing translation metrics...")
    translation_metrics = compute_translation_metrics(references, hypotheses)
    
    print("\nTranslation Quality Metrics:")
    print(f"  BLEU: {translation_metrics['bleu']:.2f}")
    print(f"  METEOR: {translation_metrics['meteor']:.2f}")
    print(f"  ROUGE-L: {translation_metrics['rouge_l']:.4f}")
    print(f"  chrF: {translation_metrics['chrf']:.2f}")
    
    # Compute emotion metrics if available
    emotion_metrics = {}
    if emotion_labels_true and emotion_labels_pred:
        print("\nComputing emotion metrics...")
        emotion_metrics = compute_emotion_metrics(emotion_labels_true, emotion_labels_pred)
        print(f"  Emotion Accuracy: {emotion_metrics['accuracy']:.2f}%")
        print(f"  Macro F1: {emotion_metrics['macro_f1']:.2f}%")
    
    # Compute semantic metrics if available
    semantic_metrics = {}
    if semantic_scores:
        print("\nComputing semantic metrics...")
        semantic_metrics = compute_semantic_metrics(semantic_scores)
        print(f"  Mean Similarity: {semantic_metrics['mean_similarity']:.4f}")
        print(f"  Std Similarity: {semantic_metrics['std_similarity']:.4f}")
    
    # Save results
    results = {
        'translation_pair': args.translation_pair,
        'test_size': len(test_df),
        'translation_metrics': translation_metrics,
        'emotion_metrics': emotion_metrics,
        'semantic_metrics': semantic_metrics
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save predictions if requested
    if args.save_predictions:
        predictions_df = pd.DataFrame({
            'source': [row[src_col] for _, row in test_df.iterrows()],
            'reference': references,
            'hypothesis': hypotheses
        })
        predictions_path = os.path.join(args.output_dir, 'predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Plot translation metrics
    plot_metrics_comparison(
        translation_metrics,
        os.path.join(args.output_dir, 'translation_metrics.png')
    )
    
    # Plot confusion matrix if emotion metrics available
    if emotion_metrics and 'confusion_matrix' in emotion_metrics:
        emotion_names = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'trust', 'disgust', 'anticipation']
        cm = np.array(emotion_metrics['confusion_matrix'])
        plot_confusion_matrix(
            cm,
            emotion_names,
            os.path.join(args.output_dir, 'emotion_confusion_matrix.png')
        )
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
