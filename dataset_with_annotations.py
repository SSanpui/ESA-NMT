"""
Fixed Dataset Class - Uses Pre-Annotated Data
This replaces the random/incorrect emotion and semantic labels
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict

class BHT25AnnotatedDataset(Dataset):
    """Dataset class that uses pre-annotated emotion and semantic labels"""

    def __init__(self, csv_path: str, tokenizer, translation_pair: str,
                 max_length: int = 128, split: str = 'train', model_type: str = 'nllb'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.translation_pair = translation_pair
        self.split = split
        self.model_type = model_type

        # Load PRE-ANNOTATED data
        self.data = self.load_annotated_data(csv_path)
        print(f"✅ Loaded {len(self.data)} ANNOTATED samples for {translation_pair} ({split})")

    def load_annotated_data(self, csv_path: str) -> List[Dict]:
        """Load pre-annotated BHT25 data"""

        # Check if already annotated or need to add _annotated suffix
        if csv_path.endswith('_annotated.csv'):
            # Already annotated file path
            annotated_path = csv_path
        else:
            # Add _annotated suffix
            annotated_path = csv_path.replace('.csv', '_annotated.csv')

        try:
            df = pd.read_csv(annotated_path)
            print(f"✅ Using ANNOTATED dataset: {annotated_path}")
        except:
            print(f"❌ Annotated dataset not found: {annotated_path}")
            print(f"⚠️  Run 'python annotate_dataset.py' first!")
            raise FileNotFoundError(f"Please run annotation script first to create {annotated_path}")

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Check required columns
        required_cols = ['bn', 'hi', 'te', 'emotion_bn', 'semantic_bn_hi', 'semantic_bn_te']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}. Please re-run annotation script.")

        # Get language pair
        from emotion_semantic_nmt_enhanced import config
        src_lang, tgt_lang = config.TRANSLATION_PAIRS[self.translation_pair]

        # Prepare data
        data = []
        for idx, row in df.iterrows():
            source_text = str(row['bn']).strip()  # Always Bengali as source
            target_text = str(row[tgt_lang]).strip()

            # Skip empty
            if len(source_text) < 3 or len(target_text) < 3:
                continue

            # Get PRE-COMPUTED annotations
            emotion_label = int(row['emotion_bn'])  # Use Bengali emotion

            # VALIDATE emotion label is in valid range (0-3 for 4 emotions)
            if emotion_label < 0 or emotion_label > 3:
                print(f"⚠️ WARNING: Invalid emotion label {emotion_label} at index {idx}")
                print(f"   Clamping to valid range [0-3]")
                emotion_label = max(0, min(3, emotion_label))  # Clamp to 0-3

            # Get semantic score based on pair
            if self.translation_pair == 'bn-hi':
                semantic_score = float(row['semantic_bn_hi'])
            elif self.translation_pair == 'bn-te':
                semantic_score = float(row['semantic_bn_te'])
            else:
                semantic_score = 0.85  # Fallback

            data.append({
                'source_text': source_text,
                'target_text': target_text,
                'emotion_label': emotion_label,  # ← REAL annotation
                'semantic_score': semantic_score,  # ← REAL semantic similarity
                'source_lang': src_lang,
                'target_lang': tgt_lang,
                'pair': self.translation_pair
            })

        print(f"Processed {len(data)} annotated samples")

        # Print annotation statistics
        print(f"\n📊 Annotation Statistics:")
        emotions = [d['emotion_label'] for d in data]
        semantics = [d['semantic_score'] for d in data]

        emotion_names = ['joy', 'sadness', 'anger', 'fear']  # MilaNLProc/xlm-emo-t outputs 4 emotions
        print(f"Emotion distribution:")
        for i in range(4):
            count = emotions.count(i)
            pct = count / len(emotions) * 100 if len(emotions) > 0 else 0
            print(f"  {emotion_names[i]:12s}: {count:4d} ({pct:5.1f}%)")

        print(f"\nSemantic similarity ({self.translation_pair}):")
        print(f"  Mean: {np.mean(semantics):.4f}")
        print(f"  Std:  {np.std(semantics):.4f}")
        print(f"  Min:  {np.min(semantics):.4f}")
        print(f"  Max:  {np.max(semantics):.4f}")

        # Split data
        if self.split == 'train':
            return data[:int(0.7 * len(data))]
        elif self.split == 'val':
            return data[int(0.7 * len(data)):int(0.85 * len(data))]
        else:  # test
            return data[int(0.85 * len(data)):]

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

        # ✅ FIX: Validate token IDs are within vocabulary range
        vocab_size = self.tokenizer.vocab_size
        source_max = source_tokens['input_ids'].max().item()
        target_max = target_tokens['input_ids'].max().item()

        if source_max >= vocab_size:
            print(f"⚠️ WARNING at idx {idx}: Source token ID {source_max} >= vocab_size {vocab_size}")
            print(f"   Source text: {source_text[:50]}...")
            source_tokens['input_ids'] = torch.clamp(source_tokens['input_ids'], 0, vocab_size - 1)

        if target_max >= vocab_size:
            print(f"⚠️ WARNING at idx {idx}: Target token ID {target_max} >= vocab_size {vocab_size}")
            print(f"   Target text: {target_text[:50]}...")
            target_tokens['input_ids'] = torch.clamp(target_tokens['input_ids'], 0, vocab_size - 1)

        # Use PRE-COMPUTED annotations (NOT random!)
        emotion_label = item['emotion_label']  # ← Real annotation
        semantic_score = item['semantic_score']  # ← Real semantic similarity
        style_label = 0  # We don't have style annotations yet (can be added later)

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
