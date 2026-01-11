import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from utils import clean_text
import random
import os

# --- CONFIGURATION ---
MODEL_NAME = "distilbert-base-uncased"

# Desactivation des logs W&B pour eviter le bruit dans le terminal
os.environ["WANDB_DISABLED"] = "true"

def check_device():
    if torch.cuda.is_available():
        print(f"Hardware Acceleration: {torch.cuda.get_device_name(0)}")
        print("Mode Haute Performance active.")
        return True
    return False

def get_big_data_dataset():
    texts = []
    labels = []

    print("\n--- PHASE 1 : Chargement et Traitement des Donnees ---")

    # ---------------------------------------------------------
    # SOURCE 1 : GONZALO A (Fake News Dataset)
    # ---------------------------------------------------------
    print("1. Chargement integral de 'gonzaloA/fake_news'...")
    try:
        ds_main = load_dataset("gonzaloA/fake_news", split='train')
        
        # Extraction des FAKES (Label 0)
        ds_fakes = ds_main.filter(lambda x: x['label'] == 0)
        print(f"   -> {len(ds_fakes)} articles FAKE recuperes.")
        
        # Extraction des REALS (Label 1)
        ds_reals_modern = ds_main.filter(lambda x: x['label'] == 1)
        print(f"   -> {len(ds_reals_modern)} articles REAL (Politique) recuperes.")

        # --- TRAITEMENT FAKE (OVERSAMPLING) ---
        # On double les fakes pour equilibrer avec la masse de vrais articles
        print("   -> Application de l'Oversampling sur les Fake News (x2)...")
        for item in ds_fakes:
            content = clean_text(str(item['title']) + " " + str(item['text']))
            # Copie 1
            texts.append(content)
            labels.append(0)
            # Copie 2 (Renforcement)
            texts.append(content)
            labels.append(0)
            
        # --- TRAITEMENT REAL (GONZALO) ---
        for item in ds_reals_modern:
            content = clean_text(str(item['title']) + " " + str(item['text']))
            texts.append(content)
            labels.append(1)

    except Exception as e:
        print(f"Erreur critique Source 1: {e}")
        return [], []

    # ---------------------------------------------------------
    # SOURCE 2 : AG NEWS (Generalist Real News)
    # ---------------------------------------------------------
    print("2. Chargement massif de 'ag_news' (Culture generale)...")
    try:
        ds_ag = load_dataset("ag_news", split='train')
        
        # On prend 30 000 articles pour equilibrer le dataset total
        count_to_take = 30000 
        
        ds_ag_subset = ds_ag.shuffle(seed=42).select(range(count_to_take))
        print(f"   -> Injection de {count_to_take} articles REAL (Science/Sport/Monde).")
        
        for item in ds_ag_subset:
            # AG News contient 'text'
            content = clean_text(str(item['text']))
            texts.append(content)
            labels.append(1) # Label 1 = REAL
            
    except Exception as e:
        print(f"Erreur Source 2: {e}")

    # ---------------------------------------------------------
    # STATISTIQUES FINALES
    # ---------------------------------------------------------
    total = len(texts)
    fakes_count = labels.count(0)
    reals_count = labels.count(1)
    
    print("\n--- BILAN DU DATASET HYBRIDE ---")
    print(f"Total articles : {total}")
    print(f"FAKE : {fakes_count}")
    print(f"REAL : {reals_count}")
    print("--------------------------------\n")
    
    # Melange integral
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts[:], labels[:] = zip(*combined)
    
    return texts, labels

def train_model():
    use_gpu = check_device()
    texts, labels = get_big_data_dataset()
    
    if len(texts) == 0: return

    print("Tokenization en cours...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Optimisation memoire : max_length=256 suffit pour la classification
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)

    class MassiveDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item
        def __len__(self):
            return len(self.labels)

    # Split 90% Train / 10% Validation
    split_idx = int(len(texts) * 0.9)
    train_dataset = MassiveDataset({k: v[:split_idx] for k, v in encodings.items()}, labels[:split_idx])
    val_dataset = MassiveDataset({k: v[split_idx:] for k, v in encodings.items()}, labels[split_idx:])

    print("Chargement du modele DistilBERT...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    
    if use_gpu: model.to('cuda')

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,              
        per_device_train_batch_size=16,  
        eval_strategy="steps",
        logging_dir='./logs',
        logging_steps=500,               
        save_total_limit=1,
        fp16=use_gpu,                    
        learning_rate=2e-5,
        save_steps=1000                  
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("\nDemarrage de l'entrainement MASSIVE DATA...")
    trainer.train()
    
    print("Sauvegarde du modele final...")
    model.save_pretrained("./fake_news_model")
    tokenizer.save_pretrained("./fake_news_model")
    print("Entrainement termine avec succes.")

if __name__ == "__main__":
    train_model()