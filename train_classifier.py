import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import clean_text
import random
import os

MODEL_NAME = "distilbert-base-cased"

os.environ["WANDB_DISABLED"] = "true"

def check_device():
    if torch.cuda.is_available():
        print(f"Hardware Acceleration: {torch.cuda.get_device_name(0)}")
        return True
    return False

def get_final_balanced_dataset():
    texts = []
    labels = []

    # Load base dataset
    print("\nPhase 1: Loading base dataset...")
    try:
        ds_main = load_dataset("gonzaloA/fake_news", split='train')
        
        ds_fakes = ds_main.filter(lambda x: x['label'] == 0)
        ds_reals = ds_main.filter(lambda x: x['label'] == 1)
        
        print(f"   FAKE articles: {len(ds_fakes)}")
        print(f"   REAL articles: {len(ds_reals)}")
        
        # Add ALL fakes (preserve conspiracy language!)
        for item in ds_fakes:
            content = clean_text(str(item['title']) + " " + str(item['text']))
            if len(content) > 50:
                texts.append(content)
                labels.append(0)
        
        # Add ALL reals from base dataset
        for item in ds_reals:
            content = clean_text(str(item['title']) + " " + str(item['text']))
            if len(content) > 50:
                texts.append(content)
                labels.append(1)
    
    except Exception as e:
        print(f"Error: {e}")
        return [], []
    
    # Count medical content
    print("\nPhase 2: Analyzing medical content...")
    medical_keywords = ['medical', 'health', 'cancer', 'cure', 'treatment', 
                       'drug', 'disease', 'doctor', 'hospital', 'pharmaceutical',
                       'Medical', 'Health', 'Cancer', 'Cure', 'Treatment']  # Include cased versions
    
    medical_fakes = 0
    medical_reals_base = 0
    
    for i, label in enumerate(labels):
        has_medical = any(kw in texts[i] for kw in medical_keywords)  # No .lower() - preserve case!
        if label == 0 and has_medical:
            medical_fakes += 1
        elif label == 1 and has_medical:
            medical_reals_base += 1
    
    print(f"   Medical FAKE: {medical_fakes}")
    print(f"   Medical REAL (base): {medical_reals_base}")
    
    # Add medical REAL articles (1.5x ratio)
    target_medical_real = int(medical_fakes * 1.5)
    need_to_add = max(0, target_medical_real - medical_reals_base)
    
    print(f"\nPhase 3: Adding medical REAL articles...")
    print(f"   Target: {target_medical_real}")
    print(f"   Need to add: {need_to_add}")
    
    if need_to_add > 0:
        print(f"   Loading {need_to_add} medical articles...")
        
        try:
            ds_ccnews = load_dataset("cc_news", split='train', streaming=True)
            
            added = 0
            for item in ds_ccnews:
                if added >= need_to_add:
                    break
                
                text = str(item.get('text', ''))
                title = str(item.get('title', ''))
                
                # Check for medical content (case-insensitive search but preserve original case)
                combined_lower = (title + " " + text).lower()
                if any(kw.lower() in combined_lower for kw in medical_keywords):
                    content = clean_text(title + " " + text)  # Preserve original case!
                    
                    # Quality filters
                    if 100 < len(content) < 3000:
                        # Must have scientific markers
                        scientific_markers = ['study', 'research', 'university', 
                                            'published', 'journal', 'trial', 'scientists',
                                            'Study', 'Research', 'University', 'Published']
                        content_lower = content.lower()
                        if sum(m.lower() in content_lower for m in scientific_markers) >= 1:
                            texts.append(content)
                            labels.append(1)
                            added += 1
                            
                            if added % 200 == 0:
                                print(f"      Progress: {added}/{need_to_add}")
            
            print(f"   Added {added} medical articles")
        
        except Exception as e:
            print(f"   Warning: {e}")
    
    # Final stats
    print("\nPhase 4: Final dataset statistics...")
    
    total = len(texts)
    final_fake = labels.count(0)
    final_real = labels.count(1)
    
    # Recount medical (case-insensitive search)
    final_medical_fake = 0
    final_medical_real = 0
    
    for i, label in enumerate(labels):
        text_lower = texts[i].lower()
        has_medical = any(kw.lower() in text_lower for kw in medical_keywords)
        if label == 0 and has_medical:
            final_medical_fake += 1
        elif label == 1 and has_medical:
            final_medical_real += 1
    
    print(f"\nFINAL DATASET:")
    print(f"   Total: {total}")
    print(f"   FAKE: {final_fake} ({final_fake/total*100:.1f}%)")
    print(f"   REAL: {final_real} ({final_real/total*100:.1f}%)")
    
    print(f"\nMEDICAL CONTENT:")
    print(f"   Medical FAKE: {final_medical_fake}")
    print(f"   Medical REAL: {final_medical_real}")
    
    if final_medical_real > 0 and final_medical_fake > 0:
        ratio = final_medical_real / final_medical_fake
        print(f"   Medical ratio: {ratio:.2f}:1")
        print("   Balanced for proper learning")
    
    print("\n" + "="*70)
    print("PREPROCESSING EXAMPLES")
    print("="*70)
    
    # Show examples of preserved signals
    for i in range(min(3, len(texts))):
        if labels[i] == 0:  # Show a fake example
            print(f"\nFAKE example (first 150 chars):")
            print(f"   {texts[i][:150]}...")
            
            # Check for signals
            signals = []
            if any(c.isupper() for c in texts[i][:150]):
                signals.append("Contains uppercase (emphasis preserved)")
            if '!' in texts[i][:150]:
                signals.append("Contains ! (emotion preserved)")
            if "'" in texts[i][:150]:
                signals.append("Contains contractions (preserved)")
            
            for signal in signals:
                print(f"   {signal}")
            break
    
    print("="*70 + "\n")
    
    # Shuffle
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts[:], labels[:] = zip(*combined)
    
    return texts, labels

def train_model():
    use_gpu = check_device()
    
    texts, labels = get_final_balanced_dataset()
    
    if len(texts) == 0:
        return
    
    print("Tokenization with CASED model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Verify tokenizer preserves case
    test_text = "BREAKING News: This is a TEST"
    test_tokens = tokenizer.tokenize(test_text)
    print(f"\nTokenizer test:")
    print(f"   Input: {test_text}")
    print(f"   Tokens: {test_tokens[:10]}")
    if any(t.isupper() or t[0].isupper() for t in test_tokens if len(t) > 0):
        print("  Tokenizer preserves case!")
    else:
        print("   Warning: Tokenizer may not preserve case well")
    
    print("\nTokenizing all texts...")
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
    
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        
        def __len__(self):
            return len(self.labels)
    
    split_idx = int(len(texts) * 0.9)
    train_dataset = NewsDataset(
        {k: v[:split_idx] for k, v in encodings.items()},
        labels[:split_idx]
    )
    val_dataset = NewsDataset(
        {k: v[split_idx:] for k, v in encodings.items()},
        labels[split_idx:]
    )
    
    print(f"Loading {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    if use_gpu:
        model.to('cuda')
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="steps",
        eval_steps=500,
        logging_steps=250,
        save_total_limit=2,
        save_steps=1000,
        fp16=use_gpu,
        learning_rate=2e-5,
        warmup_steps=500,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("\n" + "="*70)
    print("TRAINING WITH IMPROVED PREPROCESSING")
    print("Model will learn from style signals (case, punctuation, etc.)")
    print("="*70 + "\n")
    
    trainer.train()
    
    print("\nSaving model...")
    model.save_pretrained("./fake_news_model")
    tokenizer.save_pretrained("./fake_news_model")
    

if __name__ == "__main__":
    train_model()