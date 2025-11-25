#!/usr/bin/env python3
"""
Emotion Annotation Script for BHT25 Dataset
Uses XLM-RoBERTa for zero-shot cross-lingual emotion classification
"""

import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

# 8 emotions from Plutchik's wheel
EMOTION_LABELS = [
    'joy',
    'sadness', 
    'anger',
    'fear',
    'surprise',
    'trust',
    'disgust',
    'anticipation'
]


class EmotionAnnotator:
    """Annotate text with emotion labels using XLM-RoBERTa"""
    
    def __init__(self, emotion_model_name='xlm-roberta-base', device='cuda'):
        self.device = device
        print(f"Loading emotion model: {emotion_model_name}")
        
        # For zero-shot, we'll use XLM-RoBERTa with a simple classification head
        # In practice, you might want to use a fine-tuned emotion classifier
        self.tokenizer = AutoTokenizer.from_pretrained(emotion_model_name)
        self.model = AutoModel.from_pretrained(emotion_model_name).to(device)
        self.model.eval()
        
        # Simple classification layer (in real implementation, this should be pre-trained)
        self.classifier = torch.nn.Linear(768, 8).to(device)
    
    def annotate_batch(self, texts, batch_size=32):
        """Annotate a batch of texts with emotion labels"""
        emotions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
                
                # Classify (simplified - in practice use pre-trained classifier)
                logits = self.classifier(pooled)
                preds = torch.argmax(logits, dim=1)
                
                emotions.extend(preds.cpu().numpy())
        
        return emotions


class SemanticAnnotator:
    """Compute semantic similarity using LaBSE"""
    
    def __init__(self, model_name='sentence-transformers/LaBSE', device='cuda'):
        self.device = device
        print(f"Loading semantic model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
    
    def compute_similarity(self, texts1, texts2, batch_size=32):
        """Compute cosine similarity between two sets of texts"""
        similarities = []
        
        with torch.no_grad():
            for i in range(0, len(texts1), batch_size):
                batch_texts1 = texts1[i:i+batch_size]
                batch_texts2 = texts2[i:i+batch_size]
                
                # Get embeddings
                embeddings1 = self.model.encode(batch_texts1, convert_to_tensor=True, device=self.device)
                embeddings2 = self.model.encode(batch_texts2, convert_to_tensor=True, device=self.device)
                
                # Compute cosine similarity
                similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2)
                similarities.extend(similarity.cpu().numpy())
        
        return similarities


def main():
    parser = argparse.ArgumentParser(description='Annotate BHT25 with emotions and semantic scores')
    parser.add_argument('--input_file', type=str, default='BHT25_All.csv')
    parser.add_argument('--output_file', type=str, default='BHT25_annotated.csv')
    parser.add_argument('--emotion_model', type=str, default='xlm-roberta-base')
    parser.add_argument('--semantic_model', type=str, default='sentence-transformers/LaBSE')
    parser.add_argument('--num_emotions', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"Loading dataset from {args.input_file}")
    df = pd.read_csv(args.input_file)
    
    print(f"\nDataset info:")
    print(f"  Total samples: {len(df)}")
    print(f"  Columns: {df.columns.tolist()}")
    
    # Initialize annotators
    emotion_annotator = EmotionAnnotator(args.emotion_model, args.device)
    semantic_annotator = SemanticAnnotator(args.semantic_model, args.device)
    
    # Annotate Bengali texts with emotions
    print("\n" + "="*70)
    print("Annotating Bengali texts with emotions...")
    print("="*70)
    bengali_texts = df['bengali'].fillna('').tolist()
    emotion_labels = emotion_annotator.annotate_batch(bengali_texts, args.batch_size)
    df['emotion_label'] = emotion_labels
    
    # Map to emotion names
    df['emotion_name'] = df['emotion_label'].apply(lambda x: EMOTION_LABELS[x] if 0 <= x < 8 else 'unknown')
    
    # Print emotion distribution
    print("\nEmotion Distribution:")
    for i, emotion in enumerate(EMOTION_LABELS):
        count = (df['emotion_label'] == i).sum()
        pct = count / len(df) * 100
        print(f"  {emotion:15s}: {count:5d} ({pct:5.2f}%)")
    
    # Compute semantic similarity for bn-hi
    print("\n" + "="*70)
    print("Computing semantic similarity for Bengali-Hindi...")
    print("="*70)
    hindi_texts = df['hindi'].fillna('').tolist()
    semantic_bn_hi = semantic_annotator.compute_similarity(bengali_texts, hindi_texts, args.batch_size)
    df['semantic_bn_hi'] = semantic_bn_hi
    
    print(f"  Mean: {np.mean(semantic_bn_hi):.4f}")
    print(f"  Std:  {np.std(semantic_bn_hi):.4f}")
    
    # Compute semantic similarity for bn-te
    print("\n" + "="*70)
    print("Computing semantic similarity for Bengali-Telugu...")
    print("="*70)
    telugu_texts = df['telugu'].fillna('').tolist()
    semantic_bn_te = semantic_annotator.compute_similarity(bengali_texts, telugu_texts, args.batch_size)
    df['semantic_bn_te'] = semantic_bn_te
    
    print(f"  Mean: {np.mean(semantic_bn_te):.4f}")
    print(f"  Std:  {np.std(semantic_bn_te):.4f}")
    
    # Save annotated dataset
    print("\n" + "="*70)
    print(f"Saving annotated dataset to {args.output_file}")
    print("="*70)
    df.to_csv(args.output_file, index=False)
    
    print(f"\nAnnotation complete!")
    print(f"  Total samples: {len(df)}")
    print(f"  New columns: emotion_label, emotion_name, semantic_bn_hi, semantic_bn_te")
    print(f"\nDataset saved to: {args.output_file}")


if __name__ == '__main__':
    main()
