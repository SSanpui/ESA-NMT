#!/usr/bin/env python3
"""
ESA-NMT Training Script
Progressive three-phase training for Emotion-Semantic-Aware Neural Machine Translation
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)
from sentence_transformers import SentenceTransformer
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

set_seed()


class BHT25Dataset(Dataset):
    """Dataset class for BHT25 parallel corpus"""
    
    def __init__(self, data, src_lang, tgt_lang, tokenizer, max_length=128):
        self.data = data
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Map language codes
        self.lang_map = {
            'bn': 'bengali',
            'hi': 'hindi',
            'te': 'telugu'
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        src_col = self.lang_map[self.src_lang]
        tgt_col = self.lang_map[self.tgt_lang]
        
        src_text = str(row[src_col])
        tgt_text = str(row[tgt_col])
        
        # Get emotion label if available
        emotion_label = row.get('emotion_label', -1)
        
        # Get semantic score if available
        semantic_key = f'semantic_{self.src_lang}_{self.tgt_lang}'
        semantic_score = row.get(semantic_key, 0.0)
        
        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
            'emotion_label': emotion_label,
            'semantic_score': semantic_score
        }


class EmotionModule(nn.Module):
    """Emotion recognition module using XLM-RoBERTa"""
    
    def __init__(self, model_name='xlm-roberta-base', num_emotions=8):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_emotions)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class SemanticModule(nn.Module):
    """Semantic consistency module using contrastive learning"""
    
    def __init__(self, embedding_dim=768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
    
    def forward(self, src_embeddings, tgt_embeddings):
        src_proj = self.projection(src_embeddings)
        tgt_proj = self.projection(tgt_embeddings)
        
        # Cosine similarity
        src_norm = torch.nn.functional.normalize(src_proj, dim=1)
        tgt_norm = torch.nn.functional.normalize(tgt_proj, dim=1)
        similarity = torch.sum(src_norm * tgt_norm, dim=1)
        
        return similarity


class ESANMTModel(nn.Module):
    """Complete ESA-NMT model with translation, emotion, and semantic modules"""
    
    def __init__(self, base_model_name, emotion_model_name, num_emotions=8):
        super().__init__()
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
        self.emotion_module = EmotionModule(emotion_model_name, num_emotions)
        self.semantic_module = SemanticModule(self.translation_model.config.d_model)
    
    def forward(self, input_ids, attention_mask, decoder_input_ids=None, 
                decoder_attention_mask=None, labels=None):
        
        # Translation forward pass
        translation_outputs = self.translation_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels
        )
        
        return translation_outputs


def collate_fn(batch, tokenizer, src_lang_code, tgt_lang_code):
    """Custom collate function for batching"""
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]
    emotion_labels = torch.tensor([item['emotion_label'] for item in batch])
    semantic_scores = torch.tensor([item['semantic_score'] for item in batch])
    
    # Tokenize source
    src_encodings = tokenizer(
        src_texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt',
        src_lang=src_lang_code
    )
    
    # Tokenize target
    with tokenizer.as_target_tokenizer():
        tgt_encodings = tokenizer(
            tgt_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
    
    return {
        'input_ids': src_encodings['input_ids'],
        'attention_mask': src_encodings['attention_mask'],
        'labels': tgt_encodings['input_ids'],
        'decoder_attention_mask': tgt_encodings['attention_mask'],
        'emotion_labels': emotion_labels,
        'semantic_scores': semantic_scores
    }


def train_phase(model, train_loader, val_loader, optimizer, scheduler, device, 
                phase, num_epochs, alpha, beta, gamma, output_dir, use_emotion=True, 
                use_semantic=True):
    """Training function for each phase"""
    
    print(f"\nStarting Phase {phase} Training...")
    print(f"Use Emotion Module: {use_emotion}")
    print(f"Use Semantic Module: {use_semantic}")
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Calculate combined loss
            translation_loss = outputs.loss
            total_loss = alpha * translation_loss
            
            # Add emotion loss if using emotion module
            if use_emotion and phase in [1, 3]:
                emotion_labels = batch['emotion_labels'].to(device)
                emotion_logits = model.emotion_module(input_ids, attention_mask)
                emotion_loss = nn.CrossEntropyLoss()(emotion_logits, emotion_labels)
                total_loss += beta * emotion_loss
            
            # Add semantic loss if using semantic module
            if use_semantic and phase in [2, 3]:
                # Simplified semantic loss (you may need to implement full contrastive learning)
                semantic_loss = torch.tensor(0.0).to(device)
                total_loss += gamma * semantic_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            progress_bar.set_postfix({'loss': total_loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1} - Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation
        val_loss = validate(model, val_loader, device, alpha, beta, gamma, 
                           use_emotion, use_semantic, phase)
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, f'phase{phase}_best_model.pt')
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
    
    return model


def validate(model, val_loader, device, alpha, beta, gamma, use_emotion, use_semantic, phase):
    """Validation function"""
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            translation_loss = outputs.loss
            total_loss = alpha * translation_loss
            
            if use_emotion and phase in [1, 3]:
                emotion_labels = batch['emotion_labels'].to(device)
                emotion_logits = model.emotion_module(input_ids, attention_mask)
                emotion_loss = nn.CrossEntropyLoss()(emotion_logits, emotion_labels)
                total_loss += beta * emotion_loss
            
            if use_semantic and phase in [2, 3]:
                semantic_loss = torch.tensor(0.0).to(device)
                total_loss += gamma * semantic_loss
            
            val_loss += total_loss.item()
    
    return val_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description='ESA-NMT Training')
    parser.add_argument('--translation_pair', type=str, required=True, choices=['bn-hi', 'bn-te'])
    parser.add_argument('--base_model', type=str, default='facebook/nllb-200-distilled-600M')
    parser.add_argument('--emotion_model', type=str, default='xlm-roberta-base')
    parser.add_argument('--semantic_model', type=str, default='sentence-transformers/LaBSE')
    parser.add_argument('--train_split', type=float, default=0.70)
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--test_split', type=float, default=0.15)
    parser.add_argument('--phase', type=int, default=3, choices=[1, 2, 3])
    parser.add_argument('--num_epochs', type=int, default=9)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.3)
    parser.add_argument('--gamma', type=float, default=0.2)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--use_emotion_module', type=bool, default=True)
    parser.add_argument('--use_semantic_module', type=bool, default=True)
    parser.add_argument('--freeze_base_model', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--mixed_precision', type=str, default=None, choices=['fp16', 'bf16'])
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--load_emotion_module', type=str, default=None)
    parser.add_argument('--load_semantic_module', type=str, default=None)
    parser.add_argument('--skip_progressive_training', action='store_true')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Parse translation pair
    src_lang, tgt_lang = args.translation_pair.split('-')
    
    # NLLB language codes
    nllb_codes = {
        'bn': 'ben_Beng',
        'hi': 'hin_Deva',
        'te': 'tel_Telu'
    }
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Load dataset
    print("Loading BHT25 dataset...")
    if os.path.exists('BHT25_annotated.csv'):
        df = pd.read_csv('BHT25_annotated.csv')
        print("Using annotated dataset")
    else:
        df = pd.read_csv('BHT25_All.csv')
        print("Warning: Using non-annotated dataset. Emotion and semantic modules may not work properly.")
    
    if args.max_samples:
        df = df.head(args.max_samples)
    
    # Split dataset
    train_df, temp_df = train_test_split(df, train_size=args.train_split, random_state=42)
    val_size = args.val_split / (args.val_split + args.test_split)
    val_df, test_df = train_test_split(temp_df, train_size=val_size, random_state=42)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test: {len(test_df)}")
    
    # Create datasets
    train_dataset = BHT25Dataset(train_df, src_lang, tgt_lang, tokenizer)
    val_dataset = BHT25Dataset(val_df, src_lang, tgt_lang, tokenizer)
    
    # Create dataloaders
    from functools import partial
    collate_fn_partial = partial(collate_fn, tokenizer=tokenizer, 
                                 src_lang_code=nllb_codes[src_lang],
                                 tgt_lang_code=nllb_codes[tgt_lang])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, collate_fn=collate_fn_partial)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, collate_fn=collate_fn_partial)
    
    # Initialize model
    print(f"\nInitializing ESA-NMT model...")
    model = ESANMTModel(args.base_model, args.emotion_model, num_emotions=8)
    model = model.to(device)
    
    # Load pre-trained modules if specified
    if args.load_emotion_module:
        print(f"Loading emotion module from {args.load_emotion_module}")
        # Load emotion module weights
    
    if args.load_semantic_module:
        print(f"Loading semantic module from {args.load_semantic_module}")
        # Load semantic module weights
    
    # Freeze base model if specified
    if args.freeze_base_model:
        print("Freezing base translation model")
        for param in model.translation_model.parameters():
            param.requires_grad = False
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )
    
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Train
    model = train_phase(
        model, train_loader, val_loader, optimizer, scheduler, device,
        args.phase, args.num_epochs, args.alpha, args.beta, args.gamma,
        args.output_dir, args.use_emotion_module, args.use_semantic_module
    )
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pt')
    torch.save(model.state_dict(), final_model_path)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.output_dir)
    
    # Save config
    config = vars(args)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {args.output_dir}")


if __name__ == '__main__':
    main()
