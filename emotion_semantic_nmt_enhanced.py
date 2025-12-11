#!/usr/bin/env python3
"""
Enhanced Emotion-Semantic-Aware Neural Machine Translation Implementation
Comparing NLLB-200 and IndicTrans2 models with comprehensive evaluation

Features:
- NLLB-200 and IndicTrans2 model comparison
- Comprehensive metrics: BLEU, METEOR, ROUGE-L, chrF, emotion accuracy, semantic scores
- Hyperparameter tuning for alpha, beta, gamma
- Ablation study
- Separate semantic score tracking for bn-hi and bn-te
- Model deployment preparation

Author: Sudeshna Sani
"""

# ============================================================================
# 1. ENVIRONMENT SETUP AND IMPORTS
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    LogitsProcessor,
    LogitsProcessorList
)
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import json
import random
import os
import gc
import warnings
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sacrebleu
from rouge_score import rouge_scorer
from datetime import datetime

warnings.filterwarnings('ignore')

# GPU Memory Management
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️ No GPU available, using CPU")

# Environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ["WANDB_DISABLED"] = "true"

# Set device and random seeds
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# ============================================================================
# 2. CONFIGURATION
# ============================================================================

class Config:
    """Enhanced configuration with both NLLB and IndicTrans2 models"""

    # Model configurations
    MODELS = {
        'nllb': "facebook/nllb-200-distilled-600M",
        'indictrans2': "ai4bharat/indictrans2-en-indic-1B"  # Will use indic-indic variant
    }

    SEMANTIC_MODEL = "sentence-transformers/LaBSE"
    TRUST_REMOTE_CODE = True

    # Memory-optimized training parameters
    BATCH_SIZE = 2
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_LENGTH = 128

    # Architecture parameters
    HIDDEN_SIZE = 1024
    NUM_EMOTIONS = 4  # joy, sadness, anger, fear (MilaNLProc/xlm-emo-t outputs)
    PROJECTION_DIM = 256
    NUM_STYLE_TYPES = 6

    # Learning rates
    LEARNING_RATES = {
        'phase1': 5e-5,
        'phase2': 1e-5,
        'phase3': 5e-6
    }

    # Training epochs
    EPOCHS = {
        'phase1': 3,
        'phase2': 2,
        'phase3': 2
    }

    # Default loss weights (optimized for literary translation)
    ALPHA = 1.0   # Translation loss
    BETA = 0.4    # Emotion loss (increased for better emotion consistency)
    GAMMA = 0.5   # Semantic loss (INCREASED - critical for literary meaning preservation)
    DELTA = 0.1   # Style loss

    # Hyperparameter tuning ranges
    ALPHA_RANGE = [0.8, 1.0, 1.2]
    BETA_RANGE = [0.1, 0.3, 0.5]
    GAMMA_RANGE = [0.1, 0.2, 0.3]

    # Data paths
    DATA_DIR = "./data"
    OUTPUT_DIR = "./outputs"
    CHECKPOINT_DIR = "./checkpoints"
    MODELS_DIR = "./models"

    # Translation settings
    TRANSLATION_PAIRS = {
        'bn-hi': ('bn', 'hi'),
        'bn-te': ('bn', 'te'),
    }

    # Language codes
    LANG_CODES = {
        'nllb': {
            'bn': 'ben_Beng',
            'hi': 'hin_Deva',
            'te': 'tel_Telu'
        },
        'indictrans2': {
            'bn': 'ben_Beng',
            'hi': 'hin_Deva',
            'te': 'tel_Telu'
        }
    }

    # Emotion labels
    EMOTION_LABELS = ['joy', 'sadness', 'anger', 'fear', 'trust', 'disgust', 'surprise', 'anticipation']

config = Config()

# Create directories
os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.MODELS_DIR, exist_ok=True)

# ============================================================================
# 3. DATASET CLASS
# ============================================================================

class BHT25Dataset(Dataset):
    """Enhanced dataset class with emotion and semantic annotations"""

    def __init__(self, csv_path: str, tokenizer, translation_pair: str,
                 max_length: int = 128, split: str = 'train', model_type: str = 'nllb'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.translation_pair = translation_pair
        self.split = split
        self.model_type = model_type

        # Load data
        self.data = self.load_bht25_data(csv_path)
        print(f"✅ Loaded {len(self.data)} samples for {translation_pair} ({split})")

        # Load emotion classifier
        self.emotion_classifier = self.load_emotion_classifier()

        # Load semantic model for similarity
        try:
            self.semantic_model = SentenceTransformer('sentence-transformers/LaBSE')
            self.semantic_model.eval()
            if torch.cuda.is_available():
                self.semantic_model = self.semantic_model.to(device)
        except:
            self.semantic_model = None

    def load_bht25_data(self, csv_path: str) -> List[Dict]:
        """Load BHT25 CSV data"""
        try:
            df = pd.read_csv(csv_path)
            print(f"Original CSV shape: {df.shape}")

            # Clean column names
            df.columns = df.columns.str.strip().str.lower().str.replace('﻿', '')
            print(f"Columns: {df.columns.tolist()}")

            # Remove NaN
            df = df.dropna(subset=['bn', 'hi', 'te'])
            print(f"After removing NaN: {df.shape}")

            # Get language pair
            src_lang, tgt_lang = config.TRANSLATION_PAIRS[self.translation_pair]

            # Prepare data
            data = []
            for idx, row in df.iterrows():
                source_text = str(row[src_lang]).strip()
                target_text = str(row[tgt_lang]).strip()

                # Skip empty or too short/long
                if len(source_text) < 3 or len(target_text) < 3:
                    continue
                if len(source_text) > 300 or len(target_text) > 300:
                    continue

                data.append({
                    'source_text': source_text,
                    'target_text': target_text,
                    'source_lang': src_lang,
                    'target_lang': tgt_lang,
                    'pair': self.translation_pair
                })

            print(f"Processed {len(data)} valid samples")

            # Split data
            if self.split == 'train':
                return data[:int(0.7 * len(data))]
            elif self.split == 'val':
                return data[int(0.7 * len(data)):int(0.85 * len(data))]
            else:  # test
                return data[int(0.85 * len(data)):]

        except Exception as e:
            print(f"Error loading BHT25 data: {e}")
            raise

    def load_emotion_classifier(self):
        """Load emotion classifier"""
        try:
            from transformers import pipeline
            return pipeline("text-classification",
                          model="j-hartmann/emotion-english-distilroberta-base",
                          device=0 if torch.cuda.is_available() else -1,
                          top_k=None)
        except:
            return None

    def get_emotion_label(self, text: str) -> int:
        """Get emotion label for text"""
        if self.emotion_classifier is None:
            return 0

        try:
            # For multilingual text, use first 100 chars
            text_sample = text[:100]
            result = self.emotion_classifier(text_sample)

            if isinstance(result[0], list):
                result = result[0]

            # Map to our emotion labels
            emotion_map = {
                'joy': 0, 'sadness': 1, 'anger': 2, 'fear': 3,
                'trust': 4, 'disgust': 5, 'surprise': 6, 'anticipation': 7
            }

            predicted_emotion = result[0]['label'].lower()
            return emotion_map.get(predicted_emotion, 0)
        except:
            return 0

    def get_semantic_similarity(self, source: str, target: str) -> float:
        """Calculate semantic similarity between source and target"""
        if self.semantic_model is None:
            return 0.85  # Default high similarity for parallel data

        try:
            with torch.no_grad():
                embeddings = self.semantic_model.encode([source, target], convert_to_tensor=True)
                similarity = F.cosine_similarity(embeddings[0].unsqueeze(0),
                                                 embeddings[1].unsqueeze(0)).item()
            return similarity
        except:
            return 0.85

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_text = item['source_text']
        target_text = item['target_text']

        # Tokenize
        source_tokens = self.tokenizer(
            source_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        target_tokens = self.tokenizer(
            target_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Get emotion and semantic labels
        emotion_label = self.get_emotion_label(source_text)
        semantic_score = self.get_semantic_similarity(source_text, target_text)
        style_label = random.randint(0, 5)

        return {
            'source_input_ids': source_tokens['input_ids'].squeeze(),
            'source_attention_mask': source_tokens['attention_mask'].squeeze(),
            'target_input_ids': target_tokens['input_ids'].squeeze(),
            'target_attention_mask': target_tokens['attention_mask'].squeeze(),
            'emotion_label': torch.tensor(emotion_label, dtype=torch.long),
            'style_label': torch.tensor(style_label, dtype=torch.long),
            'semantic_score': torch.tensor(semantic_score, dtype=torch.float),
            'source_text': source_text,
            'target_text': target_text
        }

# ============================================================================
# 4. MODEL COMPONENTS
# ============================================================================

class EmotionModule(nn.Module):
    """Emotion recognition and integration module"""

    def __init__(self, hidden_size: int = 1024, num_emotions: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_emotions = num_emotions

        # Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_emotions)
        )

        # Emotion embeddings
        self.emotion_embedding = nn.Embedding(num_emotions, 256)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        # Projection
        self.projection = nn.Linear(256, hidden_size)

    def forward(self, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len, hidden_size = encoder_outputs.shape

        # Global pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, hidden_size)
        masked_outputs = encoder_outputs * mask_expanded
        pooled_output = masked_outputs.sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-9)

        # Emotion classification
        emotion_logits = self.emotion_classifier(pooled_output)
        emotion_probs = F.softmax(emotion_logits, dim=-1)

        # Get emotion embeddings
        emotion_ids = torch.argmax(emotion_probs, dim=-1)
        emotion_emb = self.emotion_embedding(emotion_ids)

        # Project to hidden size
        emotion_enhanced = self.projection(emotion_emb)

        # Add to encoder outputs
        enhanced_outputs = encoder_outputs + emotion_enhanced.unsqueeze(1) * 0.1

        return enhanced_outputs, emotion_logits, emotion_probs

class SemanticModule(nn.Module):
    """Semantic consistency module"""

    def __init__(self, hidden_size: int = 1024, projection_dim: int = 256):
        super().__init__()
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim

        # Projectors
        self.source_projector = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.target_projector = nn.Sequential(
            nn.Linear(hidden_size, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

        self.temperature = nn.Parameter(torch.ones([]) * 0.07)

    def forward(self, source_repr: torch.Tensor, target_repr: torch.Tensor):
        source_semantic = F.normalize(self.source_projector(source_repr), dim=-1)
        target_semantic = F.normalize(self.target_projector(target_repr), dim=-1)

        similarity = torch.cosine_similarity(source_semantic, target_semantic, dim=-1)

        return source_semantic, target_semantic, similarity

class StyleAdapter(nn.Module):
    """Style adaptation layer"""

    def __init__(self, hidden_size: int = 1024, num_style_types: int = 6):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_style_types = num_style_types

        # Style classifier
        self.style_classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_style_types)
        )

        # Adapter
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, hidden_size)
        )

    def forward(self, encoder_outputs: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len, hidden_size = encoder_outputs.shape

        # Global pooling
        mask_expanded = attention_mask.unsqueeze(-1).expand(-1, -1, hidden_size)
        masked_outputs = encoder_outputs * mask_expanded
        pooled_output = masked_outputs.sum(dim=1) / (attention_mask.sum(dim=1, keepdim=True) + 1e-9)

        # Style classification
        style_logits = self.style_classifier(pooled_output)

        # Adaptation
        adapted_outputs = self.adapter(encoder_outputs)
        final_outputs = encoder_outputs + adapted_outputs * 0.1

        return final_outputs, style_logits

# ============================================================================
# 5. MAIN MODEL
# ============================================================================

class EmotionSemanticNMT(nn.Module):
    """Emotion-Semantic-Aware NMT model with configurable components"""

    def __init__(self, config, model_type: str = 'nllb', use_emotion: bool = True,
                 use_semantic: bool = True, use_style: bool = True):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.use_emotion = use_emotion
        self.use_semantic = use_semantic
        self.use_style = use_style

        # Load base model
        print(f"Loading {model_type} model...")
        model_name = config.MODELS[model_type]
        self.base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Get hidden size
        self.hidden_size = self.base_model.config.d_model

        # Custom modules
        if use_emotion:
            self.emotion_module = EmotionModule(
                hidden_size=self.hidden_size,
                num_emotions=config.NUM_EMOTIONS
            )

        if use_semantic:
            self.semantic_module = SemanticModule(
                hidden_size=self.hidden_size,
                projection_dim=config.PROJECTION_DIM
            )

        if use_style:
            self.style_adapter = StyleAdapter(
                hidden_size=self.hidden_size,
                num_style_types=config.NUM_STYLE_TYPES
            )

        # Enable gradient checkpointing
        self.base_model.gradient_checkpointing_enable()

    def forward(self,
                source_input_ids: torch.Tensor,
                source_attention_mask: torch.Tensor,
                target_input_ids: Optional[torch.Tensor] = None,
                target_attention_mask: Optional[torch.Tensor] = None):

        if target_input_ids is not None:
            # Training mode

            # ✅ FIX: Prepare labels with proper masking
            # Mask padding positions with -100 so they're ignored in loss computation
            labels = target_input_ids[:, 1:].contiguous()

            # Mask padding tokens based on attention mask
            if target_attention_mask is not None:
                labels_attention = target_attention_mask[:, 1:]
                labels = labels.masked_fill(labels_attention == 0, -100)

            # Also explicitly mask pad_token_id (safety check)
            if self.tokenizer.pad_token_id is not None:
                labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)

            outputs = self.base_model(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                decoder_input_ids=target_input_ids[:, :-1],
                decoder_attention_mask=target_attention_mask[:, :-1] if target_attention_mask is not None else None,
                labels=labels,  # ← Use properly masked labels
                output_hidden_states=True
            )

            encoder_outputs = outputs.encoder_last_hidden_state
            decoder_outputs = outputs.decoder_hidden_states[-1] if outputs.decoder_hidden_states else None

            # Apply custom modules
            emotion_logits, emotion_probs = None, None
            style_logits = None
            semantic_similarity = None

            if self.use_emotion:
                encoder_outputs, emotion_logits, emotion_probs = self.emotion_module(
                    encoder_outputs, source_attention_mask
                )

            if self.use_style:
                encoder_outputs, style_logits = self.style_adapter(
                    encoder_outputs, source_attention_mask
                )

            if self.use_semantic and decoder_outputs is not None:
                # Get pooled representations
                src_pooled = encoder_outputs.mean(dim=1)
                tgt_pooled = decoder_outputs.mean(dim=1)
                _, _, semantic_similarity = self.semantic_module(src_pooled, tgt_pooled)

            return {
                'loss': outputs.loss,
                'logits': outputs.logits,
                'emotion_logits': emotion_logits,
                'emotion_probs': emotion_probs,
                'style_logits': style_logits,
                'semantic_similarity': semantic_similarity
            }
        else:
            # Inference mode - ENHANCED with emotion/semantic
            # First, encode the source with emotion/semantic modules
            encoder_outputs_dict = self.base_model.get_encoder()(
                input_ids=source_input_ids,
                attention_mask=source_attention_mask,
                return_dict=True
            )

            # Get encoder hidden states
            enhanced_encoder_outputs = encoder_outputs_dict.last_hidden_state

            # Apply emotion module to enhance encoder outputs
            if self.use_emotion:
                enhanced_encoder_outputs, _, _ = self.emotion_module(
                    enhanced_encoder_outputs, source_attention_mask
                )

            # Apply style adapter
            if self.use_style:
                enhanced_encoder_outputs, _ = self.style_adapter(
                    enhanced_encoder_outputs, source_attention_mask
                )

            # Create enhanced encoder outputs object
            from transformers.modeling_outputs import BaseModelOutput
            enhanced_encoder_outputs_obj = BaseModelOutput(
                last_hidden_state=enhanced_encoder_outputs,
                hidden_states=encoder_outputs_dict.hidden_states,
                attentions=encoder_outputs_dict.attentions
            )

            # Now generate using enhanced encoder outputs
            # Optimized for literary translation
            return self.base_model.generate(
                encoder_outputs=enhanced_encoder_outputs_obj,
                attention_mask=source_attention_mask,
                max_length=self.config.MAX_LENGTH,
                num_beams=5,  # Increased for better quality
                early_stopping=True,
                do_sample=False,
                length_penalty=1.0,  # Balanced length
                no_repeat_ngram_size=3  # Avoid repetition
            )

# ============================================================================
# 6. COMPREHENSIVE EVALUATOR
# ============================================================================

class ComprehensiveEvaluator:
    """Comprehensive evaluation with all metrics"""

    def __init__(self, model, tokenizer, config, translation_pair: str):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.translation_pair = translation_pair

        # Initialize scorers
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    @staticmethod
    def convert_to_json_serializable(obj):
        """Convert numpy types to Python types for JSON serialization"""
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: ComprehensiveEvaluator.convert_to_json_serializable(value)
                    for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ComprehensiveEvaluator.convert_to_json_serializable(item) for item in obj]
        else:
            return obj

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Comprehensive evaluation"""
        self.model.eval()

        # Get language pair
        src_lang, tgt_lang = config.TRANSLATION_PAIRS[self.translation_pair]
        tgt_lang_code = config.LANG_CODES[self.model.model_type][tgt_lang]

        total_loss = 0
        all_predictions = []
        all_references = []
        all_source_texts = []
        all_emotion_preds = []
        all_emotion_labels = []
        all_semantic_scores = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device)

                # Forward pass for loss
                outputs = self.model(
                    source_input_ids=batch['source_input_ids'],
                    source_attention_mask=batch['source_attention_mask'],
                    target_input_ids=batch['target_input_ids'],
                    target_attention_mask=batch['target_attention_mask']
                )

                total_loss += outputs['loss'].item()

                # Emotion accuracy
                if outputs['emotion_logits'] is not None:
                    emotion_preds = torch.argmax(outputs['emotion_logits'], dim=-1)
                    all_emotion_preds.extend(emotion_preds.cpu().numpy())
                    all_emotion_labels.extend(batch['emotion_label'].cpu().numpy())

                # Semantic scores
                if outputs['semantic_similarity'] is not None:
                    all_semantic_scores.extend(outputs['semantic_similarity'].cpu().numpy())

                # Generate translations - USE ENHANCED MODEL
                try:
                    tgt_token_id = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)

                    # Get enhanced encoder outputs with emotion/semantic modules
                    encoder_outputs_dict = self.model.base_model.get_encoder()(
                        input_ids=batch['source_input_ids'],
                        attention_mask=batch['source_attention_mask'],
                        return_dict=True
                    )

                    enhanced_encoder_outputs = encoder_outputs_dict.last_hidden_state

                    # Apply emotion module if enabled
                    if self.model.use_emotion:
                        enhanced_encoder_outputs, _, _ = self.model.emotion_module(
                            enhanced_encoder_outputs, batch['source_attention_mask']
                        )

                    # Apply style adapter if enabled
                    if self.model.use_style:
                        enhanced_encoder_outputs, _ = self.model.style_adapter(
                            enhanced_encoder_outputs, batch['source_attention_mask']
                        )

                    # Create enhanced encoder outputs object
                    from transformers.modeling_outputs import BaseModelOutput
                    enhanced_encoder_outputs_obj = BaseModelOutput(
                        last_hidden_state=enhanced_encoder_outputs,
                        hidden_states=encoder_outputs_dict.hidden_states,
                        attentions=encoder_outputs_dict.attentions
                    )

                    # Generate using enhanced encoder outputs
                    # Optimized for literary translation: diverse outputs, natural length
                    generated_ids = self.model.base_model.generate(
                        encoder_outputs=enhanced_encoder_outputs_obj,
                        attention_mask=batch['source_attention_mask'],
                        forced_bos_token_id=tgt_token_id if tgt_token_id != self.tokenizer.unk_token_id else None,
                        max_length=self.config.MAX_LENGTH,
                        num_beams=5,  # Increased from 4 for better quality
                        early_stopping=True,
                        do_sample=False,
                        length_penalty=1.0,  # Balanced length (not too short/long)
                        no_repeat_ngram_size=3,  # Avoid repetition (common in literary text)
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    predictions = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                    all_predictions.extend(predictions)
                    all_references.extend(batch['target_text'])
                    all_source_texts.extend(batch['source_text'])

                except Exception as e:
                    print(f"Generation error: {e}")
                    all_predictions.extend([""] * len(batch['target_text']))
                    all_references.extend(batch['target_text'])
                    all_source_texts.extend(batch['source_text'])

        # Calculate metrics
        metrics = {}

        # Translation metrics
        if len(all_predictions) > 0:
            # BLEU
            bleu = sacrebleu.BLEU()
            metrics['bleu'] = bleu.corpus_score(all_predictions, [all_references]).score

            # chrF
            chrf = sacrebleu.CHRF(word_order=2)
            metrics['chrf'] = chrf.corpus_score(all_predictions, [all_references]).score

            # ROUGE-L
            rouge_scores = [self.rouge_scorer.score(ref, pred)['rougeL'].fmeasure
                           for ref, pred in zip(all_references, all_predictions)]
            metrics['rouge_l'] = np.mean(rouge_scores) * 100

            # METEOR (if available)
            try:
                from nltk.translate.meteor_score import meteor_score
                meteor_scores = [meteor_score([ref.split()], pred.split())
                                for ref, pred in zip(all_references, all_predictions)]
                metrics['meteor'] = np.mean(meteor_scores) * 100
            except:
                metrics['meteor'] = 0.0

        # Emotion accuracy
        if len(all_emotion_preds) > 0:
            metrics['emotion_accuracy'] = accuracy_score(all_emotion_labels, all_emotion_preds) * 100

        # Semantic score - COMPUTE FROM GENERATED TRANSLATIONS
        # Use LaBSE to compute semantic similarity between source and generated translations
        if len(all_predictions) > 0 and len(all_source_texts) > 0:
            try:
                from sentence_transformers import SentenceTransformer
                labse_model = SentenceTransformer('sentence-transformers/LaBSE')

                # Compute embeddings for source and predictions
                source_embeddings = labse_model.encode(all_source_texts, convert_to_numpy=True, show_progress_bar=False)
                pred_embeddings = labse_model.encode(all_predictions, convert_to_numpy=True, show_progress_bar=False)

                # Compute cosine similarity
                semantic_similarities = []
                for src_emb, pred_emb in zip(source_embeddings, pred_embeddings):
                    similarity = np.dot(src_emb, pred_emb) / (np.linalg.norm(src_emb) * np.linalg.norm(pred_emb))
                    semantic_similarities.append(similarity)

                metrics['semantic_score'] = np.mean(semantic_similarities)

                # Clean up
                del labse_model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"Warning: Could not compute semantic score: {e}")
                metrics['semantic_score'] = 0.0
        else:
            metrics['semantic_score'] = 0.0

        # Loss
        metrics['avg_loss'] = total_loss / len(dataloader)
        metrics['num_samples'] = len(all_predictions)

        return metrics, all_predictions, all_references, all_source_texts

# ============================================================================
# 7. HYPERPARAMETER TUNER
# ============================================================================

class HyperparameterTuner:
    """Hyperparameter tuning for alpha, beta, gamma"""

    def __init__(self, config):
        self.config = config
        self.results = []

    def tune(self, model, train_loader, val_loader, translation_pair: str):
        """Grid search for hyperparameters"""
        print("🔍 Starting hyperparameter tuning...")

        best_score = 0
        best_params = {}

        # Grid search
        for alpha in config.ALPHA_RANGE:
            for beta in config.BETA_RANGE:
                for gamma in config.GAMMA_RANGE:
                    print(f"\nTrying alpha={alpha}, beta={beta}, gamma={gamma}")

                    # Set params
                    config.ALPHA = alpha
                    config.BETA = beta
                    config.GAMMA = gamma

                    # Train for 1 epoch
                    trainer = Trainer(model, config, translation_pair)
                    train_loss = trainer.train_epoch(train_loader, 0)

                    # Evaluate
                    evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
                    metrics, _, _, _ = evaluator.evaluate(val_loader)

                    # Combined score
                    score = metrics['bleu'] + metrics.get('emotion_accuracy', 0) * 0.5

                    self.results.append({
                        'alpha': alpha,
                        'beta': beta,
                        'gamma': gamma,
                        'bleu': metrics['bleu'],
                        'emotion_accuracy': metrics.get('emotion_accuracy', 0),
                        'score': score
                    })

                    if score > best_score:
                        best_score = score
                        best_params = {'alpha': alpha, 'beta': beta, 'gamma': gamma}

                    print(f"Score: {score:.2f} (BLEU: {metrics['bleu']:.2f})")

        # Save results
        with open(f"{config.OUTPUT_DIR}/hyperparameter_tuning_{translation_pair}.json", 'w') as f:
            json.dump(ComprehensiveEvaluator.convert_to_json_serializable({
                'results': self.results,
                'best_params': best_params,
                'best_score': best_score
            }), f, indent=2)

        print(f"\n✅ Best params: {best_params} (score: {best_score:.2f})")

        # Update config
        config.ALPHA = best_params['alpha']
        config.BETA = best_params['beta']
        config.GAMMA = best_params['gamma']

        return best_params

# ============================================================================
# 8. ABLATION STUDY
# ============================================================================

class AblationStudy:
    """Ablation study to analyze component importance"""

    def __init__(self, config):
        self.config = config
        self.results = {}

    def run(self, csv_path: str, translation_pair: str, model_type: str = 'nllb'):
        """Run ablation study"""
        print(f"🔬 Running ablation study for {translation_pair} with {model_type}...")

        configurations = [
            # Required by reviewers
            {'name': 'Baseline (No Components)', 'emotion': False, 'semantic': False, 'style': False},
            {'name': 'Emotion Only', 'emotion': True, 'semantic': False, 'style': False},
            {'name': 'Semantic Only', 'emotion': False, 'semantic': True, 'style': False},  # ← ADDED for reviewer
            {'name': 'Full Model', 'emotion': True, 'semantic': True, 'style': True},
            # Additional analysis (optional but useful)
            {'name': 'No Emotion', 'emotion': False, 'semantic': True, 'style': True},
            {'name': 'No Semantic', 'emotion': True, 'semantic': False, 'style': True},
            {'name': 'No Style', 'emotion': True, 'semantic': True, 'style': False},
        ]

        for conf in configurations:
            print(f"\n{'='*60}")
            print(f"Testing: {conf['name']}")
            print(f"{'='*60}")

            # Create model
            model = EmotionSemanticNMT(
                config,
                model_type=model_type,
                use_emotion=conf['emotion'],
                use_semantic=conf['semantic'],
                use_style=conf['style']
            ).to(device)

            # Create datasets - USE ANNOTATED for all configs (fair comparison)
            from dataset_with_annotations import BHT25AnnotatedDataset
            train_dataset = BHT25AnnotatedDataset(csv_path, model.tokenizer, translation_pair,
                                        config.MAX_LENGTH, 'train', model_type)
            val_dataset = BHT25AnnotatedDataset(csv_path, model.tokenizer, translation_pair,
                                      config.MAX_LENGTH, 'val', model_type)

            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

            # Train for 1-2 epochs
            trainer = Trainer(model, config, translation_pair)
            for epoch in range(2):
                train_loss = trainer.train_epoch(train_loader, epoch)

            # Evaluate
            evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
            metrics, _, _, _ = evaluator.evaluate(val_loader)

            self.results[conf['name']] = metrics

            print(f"\nResults for {conf['name']}:")
            print(f"  BLEU: {metrics['bleu']:.2f}")
            print(f"  chrF: {metrics['chrf']:.2f}")
            print(f"  ROUGE-L: {metrics['rouge_l']:.2f}")
            if 'emotion_accuracy' in metrics:
                print(f"  Emotion Accuracy: {metrics['emotion_accuracy']:.2f}%")
            if 'semantic_score' in metrics:
                print(f"  Semantic Score: {metrics['semantic_score']:.4f}")

            # Clean up
            del model
            torch.cuda.empty_cache()
            gc.collect()

        # Save results
        self.save_results(translation_pair, model_type)
        self.visualize_results(translation_pair, model_type)

        return self.results

    def save_results(self, translation_pair: str, model_type: str):
        """Save ablation results"""
        with open(f"{config.OUTPUT_DIR}/ablation_study_{model_type}_{translation_pair}.json", 'w') as f:
            json.dump(ComprehensiveEvaluator.convert_to_json_serializable(self.results), f, indent=2)

    def visualize_results(self, translation_pair: str, model_type: str):
        """Visualize ablation results"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        configs = list(self.results.keys())

        # BLEU scores
        bleu_scores = [self.results[c]['bleu'] for c in configs]
        axes[0, 0].barh(configs, bleu_scores, color='skyblue')
        axes[0, 0].set_xlabel('BLEU Score')
        axes[0, 0].set_title('BLEU Score by Configuration')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # chrF scores
        chrf_scores = [self.results[c]['chrf'] for c in configs]
        axes[0, 1].barh(configs, chrf_scores, color='lightgreen')
        axes[0, 1].set_xlabel('chrF Score')
        axes[0, 1].set_title('chrF Score by Configuration')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # ROUGE-L scores
        rouge_scores = [self.results[c]['rouge_l'] for c in configs]
        axes[1, 0].barh(configs, rouge_scores, color='salmon')
        axes[1, 0].set_xlabel('ROUGE-L Score')
        axes[1, 0].set_title('ROUGE-L Score by Configuration')
        axes[1, 0].grid(axis='x', alpha=0.3)

        # Combined metrics
        metrics_to_plot = ['bleu', 'chrf', 'rouge_l']
        x = np.arange(len(configs))
        width = 0.25

        for i, metric in enumerate(metrics_to_plot):
            values = [self.results[c][metric] for c in configs]
            axes[1, 1].bar(x + i*width, values, width, label=metric.upper())

        axes[1, 1].set_xlabel('Configuration')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('All Metrics Comparison')
        axes[1, 1].set_xticks(x + width)
        axes[1, 1].set_xticklabels(configs, rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{config.OUTPUT_DIR}/ablation_study_{model_type}_{translation_pair}.png",
                   dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Ablation study visualization saved!")

# ============================================================================
# 8B. INDICTRANS2 BASELINE EVALUATION
# ============================================================================

def evaluate_indictrans2_baseline(csv_path: str, translation_pair: str):
    """
    Evaluate pre-trained IndicTrans2 as baseline (no training, just evaluation)
    For reviewer comparison: NLLB vs Your Model vs IndicTrans2
    """
    print(f"\n{'='*60}")
    print(f"IndicTrans2 Baseline Evaluation: {translation_pair}")
    print(f"{'='*60}\n")

    try:
        # Load pre-trained IndicTrans2 (no custom modules)
        print("📥 Loading pre-trained IndicTrans2...")
        model = EmotionSemanticNMT(
            config,
            model_type='indictrans2',
            use_emotion=False,  # No custom modules for baseline
            use_semantic=False,
            use_style=False
        ).to(device)

        print("✅ IndicTrans2 loaded")

        # Load test dataset
        from dataset_with_annotations import BHT25AnnotatedDataset
        test_dataset = BHT25AnnotatedDataset(
            csv_path,
            model.tokenizer,
            translation_pair,
            config.MAX_LENGTH,
            'test',  # Use test split
            'indictrans2'
        )

        from torch.utils.data import DataLoader
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        print(f"📊 Evaluating on {len(test_dataset)} test samples...")

        # Evaluate
        evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
        metrics, preds, refs, sources = evaluator.evaluate(test_loader)

        # Save results
        results_file = f"{config.OUTPUT_DIR}/indictrans2_baseline_{translation_pair}.json"
        with open(results_file, 'w') as f:
            json.dump(ComprehensiveEvaluator.convert_to_json_serializable(metrics), f, indent=2)

        print(f"\n{'='*60}")
        print(f"IndicTrans2 Baseline Results:")
        print(f"{'='*60}")
        print(f"  BLEU:    {metrics['bleu']:.2f}")
        print(f"  chrF:    {metrics['chrf']:.2f}")
        print(f"  ROUGE-L: {metrics['rouge_l']:.2f}")
        print(f"\n💾 Saved to: {results_file}")

        return metrics

    except Exception as e:
        print(f"⚠️ IndicTrans2 evaluation failed: {e}")
        print("This is optional - you can skip if IndicTrans2 not available")
        return None

# ============================================================================
# 9. TRAINER
# ============================================================================

class Trainer:
    """Training manager"""

    def __init__(self, model, config, translation_pair: str):
        self.model = model
        self.config = config
        self.translation_pair = translation_pair
        self.scaler = GradScaler()

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.LEARNING_RATES['phase1'],
            weight_decay=0.01
        )

    def train_step(self, batch: Dict) -> Dict[str, float]:
        """Single training step"""
        self.model.train()

        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)

        # Forward pass
        with autocast():
            outputs = self.model(
                source_input_ids=batch['source_input_ids'],
                source_attention_mask=batch['source_attention_mask'],
                target_input_ids=batch['target_input_ids'],
                target_attention_mask=batch['target_attention_mask']
            )

            # Calculate losses
            translation_loss = outputs['loss']
            total_loss = self.config.ALPHA * translation_loss

            if outputs['emotion_logits'] is not None:
                emotion_loss = F.cross_entropy(outputs['emotion_logits'], batch['emotion_label'])
                total_loss += self.config.BETA * emotion_loss

            if outputs['style_logits'] is not None:
                style_loss = F.cross_entropy(outputs['style_logits'], batch['style_label'])
                total_loss += self.config.DELTA * style_loss

            if outputs['semantic_similarity'] is not None:
                semantic_loss = F.mse_loss(outputs['semantic_similarity'], batch['semantic_score'])
                total_loss += self.config.GAMMA * semantic_loss

            # Scale for gradient accumulation
            total_loss = total_loss / self.config.GRADIENT_ACCUMULATION_STEPS

        # Backward pass
        self.scaler.scale(total_loss).backward()

        return {
            'total_loss': total_loss.item() * self.config.GRADIENT_ACCUMULATION_STEPS,
            'translation_loss': translation_loss.item()
        }

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Train one epoch"""
        total_loss = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for step, batch in enumerate(progress_bar):
            losses = self.train_step(batch)
            total_loss += losses['total_loss']

            # Update weights
            if (step + 1) % self.config.GRADIENT_ACCUMULATION_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Update progress
            progress_bar.set_postfix({
                'loss': f'{losses["total_loss"]:.4f}',
                'mem': f'{torch.cuda.memory_allocated()/1024**3:.1f}GB' if torch.cuda.is_available() else 'N/A'
            })

            if (step + 1) % 50 == 0:
                torch.cuda.empty_cache()

        return total_loss / len(dataloader)

# ============================================================================
# 10. MODEL COMPARISON
# ============================================================================

def compare_models(csv_path: str, translation_pair: str):
    """Compare NLLB and IndicTrans2 models"""
    print(f"\n{'='*60}")
    print(f"MODEL COMPARISON: {translation_pair.upper()}")
    print(f"{'='*60}\n")

    results = {}

    for model_type in ['nllb']:  # Add 'indictrans2' when available
        print(f"\n🔄 Training {model_type.upper()} model...")

        # Create model
        model = EmotionSemanticNMT(config, model_type=model_type).to(device)

        # Create datasets
        train_dataset = BHT25Dataset(csv_path, model.tokenizer, translation_pair,
                                    config.MAX_LENGTH, 'train', model_type)
        val_dataset = BHT25Dataset(csv_path, model.tokenizer, translation_pair,
                                  config.MAX_LENGTH, 'val', model_type)
        test_dataset = BHT25Dataset(csv_path, model.tokenizer, translation_pair,
                                   config.MAX_LENGTH, 'test', model_type)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        # Train
        trainer = Trainer(model, config, translation_pair)
        for epoch in range(config.EPOCHS['phase1']):
            train_loss = trainer.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")

        # Evaluate
        evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
        metrics, preds, refs, sources = evaluator.evaluate(test_loader)

        results[model_type] = metrics

        print(f"\n{model_type.upper()} Results:")
        print(f"  BLEU: {metrics['bleu']:.2f}")
        print(f"  chrF: {metrics['chrf']:.2f}")
        print(f"  ROUGE-L: {metrics['rouge_l']:.2f}")
        if 'emotion_accuracy' in metrics:
            print(f"  Emotion Accuracy: {metrics['emotion_accuracy']:.2f}%")
        if 'semantic_score' in metrics:
            print(f"  Semantic Score: {metrics['semantic_score']:.4f}")

        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'metrics': metrics
        }, f"{config.CHECKPOINT_DIR}/best_model_{model_type}_{translation_pair}.pt")

        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Save comparison results
    with open(f"{config.OUTPUT_DIR}/model_comparison_{translation_pair}.json", 'w') as f:
        json.dump(ComprehensiveEvaluator.convert_to_json_serializable(results), f, indent=2)

    # Visualize comparison
    visualize_model_comparison(results, translation_pair)

    return results

def visualize_model_comparison(results: Dict, translation_pair: str):
    """Visualize model comparison"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = list(results.keys())
    metrics = ['bleu', 'chrf', 'rouge_l']

    # Bar chart
    x = np.arange(len(metrics))
    width = 0.35

    for i, model in enumerate(models):
        values = [results[model][m] for m in metrics]
        axes[0].bar(x + i*width, values, width, label=model.upper())

    axes[0].set_xlabel('Metric')
    axes[0].set_ylabel('Score')
    axes[0].set_title(f'Model Comparison: {translation_pair.upper()}')
    axes[0].set_xticks(x + width/2)
    axes[0].set_xticklabels([m.upper() for m in metrics])
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # Radar chart (if multiple models)
    if len(models) > 1:
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        ax = plt.subplot(122, projection='polar')

        for model in models:
            values = [results[model][m] for m in metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model.upper())
            ax.fill(angles, values, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_title(f'Performance Radar: {translation_pair.upper()}')
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/model_comparison_{translation_pair}.png",
               dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 11. DEPLOYMENT
# ============================================================================

def prepare_for_deployment(model, model_type: str, translation_pair: str):
    """Prepare model for deployment to Hugging Face"""
    print(f"💾 Preparing {model_type} model for deployment...")

    output_dir = f"{config.MODELS_DIR}/emotion-semantic-nmt-{model_type}-{translation_pair}"
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model.base_model.save_pretrained(output_dir)
    model.tokenizer.save_pretrained(output_dir)

    # Save custom modules
    torch.save({
        'emotion_module': model.emotion_module.state_dict() if model.use_emotion else None,
        'semantic_module': model.semantic_module.state_dict() if model.use_semantic else None,
        'style_adapter': model.style_adapter.state_dict() if model.use_style else None,
        'config': config
    }, f"{output_dir}/custom_modules.pt")

    # Create README
    readme_content = f"""---
language:
- bn
- hi
- te
license: mit
tags:
- translation
- emotion
- semantic
- {translation_pair}
---

# Emotion-Semantic-Aware NMT: {translation_pair.upper()}

This model performs emotion-semantic-aware neural machine translation between {translation_pair.replace('-', ' and ')}.

## Model Details

- **Base Model**: {model_type}
- **Translation Pair**: {translation_pair}
- **Components**: Emotion Module, Semantic Module, Style Adapter

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("emotion-semantic-nmt-{model_type}-{translation_pair}")
model = AutoModelForSeq2SeqLM.from_pretrained("emotion-semantic-nmt-{model_type}-{translation_pair}")

# Translate
text = "Your text here"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs)
translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Citation

```bibtex
@article{{emotion-semantic-nmt,
  title={{Emotion-Semantic-Aware Neural Machine Translation for Indo-Aryan and Dravidian Languages}},
  author={{Sudeshna Sani}},
  year={{2025}}
}}
```
"""

    with open(f"{output_dir}/README.md", 'w') as f:
        f.write(readme_content)

    print(f"✅ Model saved to {output_dir}")
    print(f"📤 Ready for Hugging Face upload!")

    return output_dir

# ============================================================================
# 12. MAIN EXECUTION
# ============================================================================

# ============================================================================
# PROGRAMMATIC API (for Colab notebook)
# ============================================================================

def full_training_pipeline(csv_path: str, translation_pair: str, model_type: str = 'nllb'):
    """
    Full training pipeline - can be called from notebook without interactive input

    Args:
        csv_path: Path to dataset CSV
        translation_pair: 'bn-hi' or 'bn-te'
        model_type: 'nllb' or 'indictrans2'
    """
    from dataset_with_annotations import BHT25AnnotatedDataset

    print(f"\n🚀 Starting Full Training Pipeline")
    print(f"   Translation: {translation_pair}")
    print(f"   Model: {model_type}")
    print(f"   Epochs: {config.EPOCHS['phase1']}")
    print("="*60)

    # Create model
    print("\n1️⃣ Creating model...")
    model = EmotionSemanticNMT(config, model_type=model_type).to(device)
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load ANNOTATED dataset
    print("\n2️⃣ Loading annotated dataset...")
    train_dataset = BHT25AnnotatedDataset(csv_path, model.tokenizer, translation_pair,
                                config.MAX_LENGTH, 'train', model_type)
    val_dataset = BHT25AnnotatedDataset(csv_path, model.tokenizer, translation_pair,
                              config.MAX_LENGTH, 'val', model_type)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")

    # Train
    print(f"\n3️⃣ Training for {config.EPOCHS['phase1']} epochs...")
    trainer = Trainer(model, config, translation_pair)

    for epoch in range(config.EPOCHS['phase1']):
        print(f"\n--- Epoch {epoch+1}/{config.EPOCHS['phase1']} ---")
        train_loss = trainer.train_epoch(train_loader, epoch)
        print(f"Train Loss: {train_loss:.4f}")

        # Evaluate
        if (epoch + 1) % 1 == 0:
            evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
            metrics, _, _, _ = evaluator.evaluate(val_loader)
            print(f"Validation - BLEU: {metrics['bleu']:.2f}, chrF: {metrics['chrf']:.2f}, "
                  f"Emotion Acc: {metrics.get('emotion_accuracy', 0):.2f}%")

    # Final evaluation
    print("\n4️⃣ Final Evaluation...")
    evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
    metrics, preds, refs, sources = evaluator.evaluate(val_loader)

    print("\n📊 Final Results:")
    print("="*60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"   {key:20s}: {value:.4f}")
        else:
            print(f"   {key:20s}: {value}")

    # Save model
    print("\n5️⃣ Saving model...")
    checkpoint_path = f"{config.CHECKPOINT_DIR}/final_model_{model_type}_{translation_pair}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'metrics': metrics
    }, checkpoint_path)
    print(f"   Model saved: {checkpoint_path}")

    # Save results
    results_path = f"{config.OUTPUT_DIR}/full_training_results_{model_type}_{translation_pair}.json"
    with open(results_path, 'w') as f:
        import json
        json.dump(ComprehensiveEvaluator.convert_to_json_serializable(metrics), f, indent=2)
    print(f"   Results saved: {results_path}")

    print("\n✅ Full training completed!")
    return metrics


def main():
    """Main execution function"""
    print("""
╔════════════════════════════════════════════════════════════╗
║  Enhanced Emotion-Semantic-Aware Neural Machine Translation ║
║  NLLB-200 vs IndicTrans2 Comparison                       ║
╚════════════════════════════════════════════════════════════╝
""")

    csv_path = "BHT25_All.csv"

    print("\nMenu:")
    print("1. Compare models (NLLB vs IndicTrans2)")
    print("2. Run ablation study")
    print("3. Hyperparameter tuning")
    print("4. Train specific model")
    print("5. Evaluate model")
    print("6. Prepare for deployment")

    choice = input("\nEnter choice (1-6): ").strip()
    translation_pair = input("Enter translation pair (bn-hi/bn-te): ").strip() or 'bn-hi'

    if choice == '1':
        # Compare models
        results = compare_models(csv_path, translation_pair)

    elif choice == '2':
        # Ablation study
        model_type = input("Enter model type (nllb/indictrans2): ").strip() or 'nllb'
        ablation = AblationStudy(config)
        ablation.run(csv_path, translation_pair, model_type)

    elif choice == '3':
        # Hyperparameter tuning
        model_type = input("Enter model type (nllb/indictrans2): ").strip() or 'nllb'
        model = EmotionSemanticNMT(config, model_type=model_type).to(device)

        train_dataset = BHT25Dataset(csv_path, model.tokenizer, translation_pair,
                                    config.MAX_LENGTH, 'train', model_type)
        val_dataset = BHT25Dataset(csv_path, model.tokenizer, translation_pair,
                                  config.MAX_LENGTH, 'val', model_type)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        tuner = HyperparameterTuner(config)
        best_params = tuner.tune(model, train_loader, val_loader, translation_pair)

        print(f"\n✅ Best hyperparameters: {best_params}")

    elif choice == '4':
        # Train specific model
        model_type = input("Enter model type (nllb/indictrans2): ").strip() or 'nllb'
        model = EmotionSemanticNMT(config, model_type=model_type).to(device)

        train_dataset = BHT25Dataset(csv_path, model.tokenizer, translation_pair,
                                    config.MAX_LENGTH, 'train', model_type)
        val_dataset = BHT25Dataset(csv_path, model.tokenizer, translation_pair,
                                  config.MAX_LENGTH, 'val', model_type)

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

        trainer = Trainer(model, config, translation_pair)

        for epoch in range(config.EPOCHS['phase1']):
            train_loss = trainer.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}")

            if (epoch + 1) % 1 == 0:
                evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
                metrics, _, _, _ = evaluator.evaluate(val_loader)
                print(f"Validation - BLEU: {metrics['bleu']:.2f}, chrF: {metrics['chrf']:.2f}")

        # Save
        torch.save(model.state_dict(),
                  f"{config.CHECKPOINT_DIR}/final_model_{model_type}_{translation_pair}.pt")

    elif choice == '5':
        # Evaluate
        model_type = input("Enter model type (nllb/indictrans2): ").strip() or 'nllb'
        model = EmotionSemanticNMT(config, model_type=model_type).to(device)

        checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model_{model_type}_{translation_pair}.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            test_dataset = BHT25Dataset(csv_path, model.tokenizer, translation_pair,
                                       config.MAX_LENGTH, 'test', model_type)
            test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

            evaluator = ComprehensiveEvaluator(model, model.tokenizer, config, translation_pair)
            metrics, preds, refs, sources = evaluator.evaluate(test_loader)

            print(f"\nTest Results:")
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

    elif choice == '6':
        # Prepare for deployment
        model_type = input("Enter model type (nllb/indictrans2): ").strip() or 'nllb'
        model = EmotionSemanticNMT(config, model_type=model_type).to(device)

        checkpoint_path = f"{config.CHECKPOINT_DIR}/best_model_{model_type}_{translation_pair}.pt"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])

            output_dir = prepare_for_deployment(model, model_type, translation_pair)
            print(f"\n✅ Model ready for deployment at: {output_dir}")
        else:
            print(f"No checkpoint found at {checkpoint_path}")

if __name__ == "__main__":
    main()
