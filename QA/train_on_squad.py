"""
Training script rapide sur the dataset SQuAD 2.0.
Permet de tester l'approche Q&A avec un dataset classique.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import os

# Configuration
MODEL_NAME = "google/flan-t5-base"  # Déjà pré-entraîné mais on peut affiner
OUTPUT_DIR = "./qa_model_squad"
os.environ["WANDB_DISABLED"] = "true"

class SQuADTrainer:
    def __init__(self, model_name=MODEL_NAME, max_samples=None):
        """
        Trainer pour SQuAD 2.0.
        
        Args:
            model_name: Modèle de base à utiliser
            max_samples: Limite the nombre d'examples (pour tests rapides)
        """
        self.model_name = model_name
        self.max_samples = max_samples
        self.output_dir = OUTPUT_DIR
        
        print("="*80)
        print("TRAINING ON SQUAD 2.0")
        print("="*80)
        
        # Check GPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            print(f" GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print(" CPU only (slower)")
        
        # Load tokenizer et model
        print(f"\nLoading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    def load_squad(self):
        """
        Charge the dataset SQuAD 2.0.
        """
        print("\nLoading SQuAD 2.0...")
        
        # Load the dataset complet
        dataset = load_dataset("rajpurkar/squad_v2")
        
        # Limit si demandé (pour tests rapides)
        if self.max_samples:
            print(f" Mode test: limitation à {self.max_samples} examples")
            train_dataset = dataset['train'].select(range(min(self.max_samples, len(dataset['train']))))
            val_dataset = dataset['validation'].select(range(min(self.max_samples//10, len(dataset['validation']))))
        else:
            train_dataset = dataset['train']
            val_dataset = dataset['validation']
        
        print(f" Train: {len(train_dataset)} examples")
        print(f" Validation: {len(val_dataset)} examples")
        
        # Statistics
        answerable_train = sum(1 for ex in train_dataset if ex['answers']['text'])
        answerable_val = sum(1 for ex in val_dataset if ex['answers']['text'])
        
        print(f"\nDistribution:")
        print(f"  Train - Answerable: {answerable_train}/{len(train_dataset)} ({answerable_train/len(train_dataset)*100:.1f}%)")
        print(f"  Val   - Answerable: {answerable_val}/{len(val_dataset)} ({answerable_val/len(val_dataset)*100:.1f}%)")
        
        return train_dataset, val_dataset
    
    def preprocess_function(self, examples):
        """
        Prétraite the examples SQuAD pour T5.
        Format: "question: <Q> context: <C>" -> "<Answer>" ou "unanswerable"
        """
        # Préparer the inputs
        inputs = []
        targets = []
        
        for i in range(len(examples['question'])):
            question = examples['question'][i]
            context = examples['context'][i]
            
            # Input format
            input_text = f"question: {question} context: {context}"
            inputs.append(input_text)
            
            # Target (answer ou "unanswerable")
            answers = examples['answers'][i]
            if answers['text']:
                # Question answerable
                target = answers['text'][0]
            else:
                # Question unanswerable
                target = "unanswerable"
            
            targets.append(target)
        
        # Tokenize inputs
        model_inputs = self.tokenizer(
            inputs,
            max_length=512,
            truncation=True,
            padding=False
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
    
    def train(self, num_epochs=3, batch_size=8, learning_rate=3e-5, quick_test=False):
        """
        Trains the model on SQuAD.
        
        Args:
            num_epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            quick_test: If True, quick test mode (1000 examples, 1 epoch)
        """
        # Reduce batch_size for quick_test to avoid OOM on smaller GPUs
        if quick_test:
            batch_size = 2
        
        # Quick test mode
        if quick_test:
            print("\n QUICK TEST MODE ACTIVATED")
            print("   - 1000 training examples")
            print("   - 1 epoch")
            print("   - For complete test, launch without quick_test=True")
            self.max_samples = 1000
            num_epochs = 1
        
        # Load data
        train_dataset, val_dataset = self.load_squad()
        
        # Preprocess
        print("\nPreprocessing data...")
        train_dataset = train_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names,
            desc="Preprocessing train"
        )
        
        val_dataset = val_dataset.map(
            self.preprocess_function,
            batched=True,
            remove_columns=val_dataset.column_names,
            desc="Preprocessing validation"
        )
        
        print(f" Preprocessing complete")
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            
            # Hyperparameters
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=0.01,
            
            # Warmup
            warmup_steps=500,
            
            # Evaluation
            eval_strategy="steps",
            eval_steps=500 if not quick_test else 100,
            logging_steps=100 if not quick_test else 50,
            save_steps=500 if not quick_test else 100,
            save_total_limit=2,
            
            # Optimizations
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            
            # Best model
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Generation
            predict_with_generate=True,
            generation_max_length=150,
            
            # Other
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
        print("\n" + "="*80)
        print("TRAINING START")
        print("="*80)
        
        if quick_test:
            print("\n Quick test mode - short training")
        else:
            print(f"\n Configuration:")
            print(f"   - Epochs: {num_epochs}")
            print(f"   - Batch size: {batch_size}")
            print(f"   - Learning rate: {learning_rate}")
            print(f"   - Training examples: {len(train_dataset)}")
            print(f"   - Validation examples: {len(val_dataset)}")
        
        print("\n Training in progress...\n")
        
        trainer.train()
        
        # Save
        print(f"\n Saving the model dans {self.output_dir}")
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Métriques finales
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        
        final_metrics = trainer.evaluate()
        print("\nFinal metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return trainer
    
    def test_model(self, num_examples=5):
        """
        Teste the model entraîné sur quelques examples.
        """
        print("\n" + "="*80)
        print("MODEL TEST")
        print("="*80)
        
        # Load the model entraîné
        print(f"\nLoading model from {self.output_dir}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(self.output_dir)
        tokenizer = AutoTokenizer.from_pretrained(self.output_dir)
        model.to(self.device)
        model.eval()
        
        # Load examples de test
        print("Loading test examples...")
        test_data = load_dataset("rajpurkar/squad_v2", split=f"validation[:{num_examples}]")
        
        print("\n" + "-"*80)
        print("PREDICTION EXAMPLES")
        print("-"*80)
        
        for i, example in enumerate(test_data):
            question = example['question']
            context = example['context']
            
            # Prédire
            input_text = f"question: {question} context: {context}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=150, num_beams=4)
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Display
            print(f"\n[Exemple {i+1}]")
            print(f"Context (excerpt): {context[:200]}...")
            print(f"Question: {question}")
            print(f"Prediction: {prediction}")
            
            if example['answers']['text']:
                print(f"Reference: {example['answers']['text'][0]}")
            else:
                print(f"Reference: [UNANSWERABLE]")
            
            print("-" * 80)

def main():
    """
    Fonction principale.
    """
    import sys
    
    print("="*80)
    print("QUICK TRAINING ON SQUAD 2.0")
    print("="*80)
    
    print("\nAvailable options:")
    print("  1. Quick test (1000 examples, 1 epoch) - ~5-10 min")
    print("  2. Complete training (130K examples, 3 epochs) - ~2-3h GPU")
    print("  3. Medium training (10K examples, 2 epochs) - ~30 min")
    
    try:
        choice = input("\nYour choice (1/2/3): ").strip()
    except:
        choice = "1"
    
    trainer = SQuADTrainer()
    
    if choice == "1":
        # Test rapide
        print("\n Starting quick test...")
        trainer.train(quick_test=True)
        trainer.test_model(num_examples=5)
        
    elif choice == "2":
        # Complet
        print("\n Starting complete training...")
        print("   (This may take 2-3 hours on GPU)")
        confirm = input("   Continue? (y/n): ").strip().lower()
        if confirm == 'y':
            trainer.train(num_epochs=3, batch_size=8)
            trainer.test_model(num_examples=10)
        else:
            print("Canceled.")
            return
    
    elif choice == "3":
        # Moyen
        print("\n Starting medium training...")
        trainer.max_samples = 10000
        trainer.train(num_epochs=2, batch_size=8)
        trainer.test_model(num_examples=8)
    
    else:
        print("Invalid choice. Starting quick test by default...")
        trainer.train(quick_test=True)
        trainer.test_model(num_examples=5)
    
    print("\n" + "="*80)
    print(" COMPLETE")
    print("="*80)
    print(f"\nModèle saved dans: {OUTPUT_DIR}")
if __name__ == "__main__":
    main()