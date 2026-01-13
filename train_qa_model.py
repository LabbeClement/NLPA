"""
Training script pour the model Q&A.
Fine-tune un model T5/BART sur the dataset Q&A généré.
"""

import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import numpy as np
import os

# Choix the model of base
MODEL_NAME = "google/flan-t5-base"  # tester avec ca : flan-t5-base, flan-t5-large (bcp trop lourd pc PLS), bart-base (a tester)
os.environ["WANDB_DISABLED"] = "true"

class QATrainer:
    def __init__(self, model_name=MODEL_NAME):
        """
        Initialise the trainer pour the model Q&A.
        """
        self.model_name = model_name
        self.output_dir = "./qa_model_finetuned"
        
        print(f"Initializing the trainer Q&A avec {model_name}")
        
        # Check the GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f" GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(" CPU uniquement (training plus lent)")
        
        # Load tokenizer et model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def load_dataset(self, train_file, val_file):
        """
        Charge the datasets train et validation depuis the files JSON.
        """
        print(f"\nLoading datasets...")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        with open(val_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        print(f"   Train: {len(train_data)} examples")
        print(f"   Validation: {len(val_data)} examples")
        
        # Convertir en Hugging Face Dataset
        train_dataset = Dataset.from_dict({
            'context': [item['context'] for item in train_data],
            'question': [item['question'] for item in train_data],
            'answer': [item['answer'] for item in train_data],
            'question_type': [item['question_type'] for item in train_data],
            'label': [item['label'] for item in train_data]
        })
        
        val_dataset = Dataset.from_dict({
            'context': [item['context'] for item in val_data],
            'question': [item['question'] for item in val_data],
            'answer': [item['answer'] for item in val_data],
            'question_type': [item['question_type'] for item in val_data],
            'label': [item['label'] for item in val_data]
        })
        
        return train_dataset, val_dataset
    
    def preprocess_function(self, examples):
        """
        Prétraite the examples for training seq2seq.
        Format: "question: <Q> context: <C>" -> "<Answer>"
        """
        # Input: question + context
        inputs = [
            f"question: {q} context: {c}" 
            for q, c in zip(examples['question'], examples['context'])
        ]
        
        # Target: answer
        targets = examples['answer']
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=False  # Le data collator s'en chargera
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            targets,
            max_length=150,
            truncation=True,
            padding=False
        )
        
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def train(self, train_file="qa_dataset_train.json", val_file="qa_dataset_validation.json"):
        """
        Trains the model Q&A.
        """
        print("\n" + "="*80)
        print("TRAINING DU MODEL Q&A")
        print("="*80)
        
        # Load the data
        train_dataset, val_dataset = self.load_dataset(train_file, val_file)
        
        # Prétraiter
        print("\nTokenization...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Tokenizing train"
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Tokenizing validation"
        )
        
        # Data collator pour the padding dynamique
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Arguments of training
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            
            # Hyperparamètres principaux
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=16,
            learning_rate=3e-5,
            weight_decay=0.01,
            
            # Warmup
            warmup_steps=500,
            
            # Evaluation et logging
            eval_strategy="steps",
            eval_steps=500,
            logging_steps=100,
            save_steps=500,
            save_total_limit=3,
            
            # Optimisations
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            
            # Meilleur model
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Generating pour evaluation
            predict_with_generate=True,
            generation_max_length=150,
            
            # Autres
            report_to="none",
            push_to_hub=False,
        )
        
        # Create the trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Entraîner
        print("\ntraining...\n")
        trainer.train()
        
        # Save the model final
        print(f"\n Saving the model dans {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Display the métriques finales
        print("\n" + "="*80)
        print("TRAINING TERMINÉ")
        print("="*80)
        
        final_metrics = trainer.evaluate()
        print("\nMétriques finales:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return trainer

def main():
    """
    Fonction principale pour entraîner the model Q&A.
    """
    # Create the trainer
    trainer = QATrainer(model_name=MODEL_NAME)
    
    # Check que the files of dataset existent
    train_file = "qa_dataset_train.json"
    val_file = "qa_dataset_validation.json"
    
    if not os.path.exists(train_file) or not os.path.exists(val_file):
        print(f"   Fichiers requis:")
        print(f"     - {train_file}")
        print(f"     - {val_file}")
        return
    
    # Entraîner
    trainer.train(train_file=train_file, val_file=val_file)
if __name__ == "__main__":
    main()