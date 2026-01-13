"""
Evaluation script the model sur SQuAD 2.0.
Calcule the métriques standard: F1 et Exact Match.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import string
import re

def normalize_answer(s):
    """
    Normalise une answer pour the calcul of EM et F1.
    (Standard SQuAD)
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, ground_truth):
    """
    Calcule Exact Match (EM).
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    """
    Calcule F1 score token-level.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # Si the deux sont vides
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return 1.0
    
    # Si l'un est vide
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return 0.0
    
    common = set(pred_tokens) & set(truth_tokens)
    num_common = len(common)
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(truth_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

class SQuADEvaluator:
    def __init__(self, model_path="./qa_model_squad"):
        """
        Évaluateur pour model SQuAD.
        """
        print(f"\nLoading the model depuis {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        print(f" Model loaded on {self.device}")
    
    def generate_answer(self, question, context):
        """
        Generates an answer.
        """
        input_text = f"question: {question} context: {context}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=4,
                early_stopping=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer
    
    def evaluate(self, num_examples=None, split="validation"):
        """
        Evaluates the model sur SQuAD.
        
        Args:
            num_examples: Nombre of examples à évaluer (None = tous)
            split: Split à utiliser (validation ou train)
        """
        print(f"\nLoading dataset SQuAD ({split})...")
        
        if num_examples:
            dataset = load_dataset("rajpurkar/squad_v2", split=f"{split}[:{num_examples}]")
            print(f" {len(dataset)}")
        else:
            dataset = load_dataset("rajpurkar/squad_v2", split=split)
            print(f" {len(dataset)}")
        
        # Métriques
        exact_matches = []
        f1_scores = []
        
        # Métriques pour answerable vs unanswerable
        answerable_em = []
        answerable_f1 = []
        unanswerable_em = []
        unanswerable_f1 = []
        
        print("\nEvaluation in progress...")
        
        for example in tqdm(dataset, desc="Evaluating"):
            question = example['question']
            context = example['context']
            
            # Generate prédiction
            prediction = self.generate_answer(question, context)
            
            # Ground truth
            if example['answers']['text']:
                # Question answerable
                is_answerable = True
                ground_truth = example['answers']['text'][0]
            else:
                # Question unanswerable
                is_answerable = False
                ground_truth = "unanswerable"
            
            # Calculate métriques
            em = compute_exact_match(prediction, ground_truth)
            f1 = compute_f1(prediction, ground_truth)
            
            exact_matches.append(em)
            f1_scores.append(f1)
            
            # Séparer answerable/unanswerable
            if is_answerable:
                answerable_em.append(em)
                answerable_f1.append(f1)
            else:
                unanswerable_em.append(em)
                unanswerable_f1.append(f1)
        
        # Résultats
        results = {
            'overall': {
                'exact_match': np.mean(exact_matches),
                'f1': np.mean(f1_scores),
                'num_examples': len(dataset)
            },
            'answerable': {
                'exact_match': np.mean(answerable_em) if answerable_em else 0.0,
                'f1': np.mean(answerable_f1) if answerable_f1 else 0.0,
                'num_examples': len(answerable_em)
            },
            'unanswerable': {
                'exact_match': np.mean(unanswerable_em) if unanswerable_em else 0.0,
                'f1': np.mean(unanswerable_f1) if unanswerable_f1 else 0.0,
                'num_examples': len(unanswerable_em)
            }
        }
        
        # Affichage
        self._print_results(results)
        
        return results
    
    def _print_results(self, results):
        """
        Affiche the results of manière formatée.
        """
        print("RESULTS D'EVALUATION")

        # Overall
        print("\n RESULTS GLOBAUX")
        print(f"{'─'*40}")
        print(f"Exact Match:  {results['overall']['exact_match']:.2%}")
        print(f"F1 Score:     {results['overall']['f1']:.2%}")
        print(f"Examples:     {results['overall']['num_examples']}")
        
        # Answerable
        print("\n QUESTIONS AVEC ANSWER")
        print(f"{'─'*40}")
        print(f"Exact Match:  {results['answerable']['exact_match']:.2%}")
        print(f"F1 Score:     {results['answerable']['f1']:.2%}")
        print(f"Examples:     {results['answerable']['num_examples']}")
        
        # Unanswerable
        print("\n QUESTIONS SANS ANSWER")
        print(f"{'─'*40}")
        print(f"Exact Match:  {results['unanswerable']['exact_match']:.2%}")
        print(f"F1 Score:     {results['unanswerable']['f1']:.2%}")
        print(f"Examples:     {results['unanswerable']['num_examples']}")
    
    
    def compare_examples(self, num_examples=10):
        """
        Compare examples of prédictions.
        """
        print("\n" + "="*80)
        print("COMPARAISON D'EXAMPLES")
        print("="*80)
        
        dataset = load_dataset("rajpurkar/squad_v2", split=f"validation[:{num_examples}]")
        
        for i, example in enumerate(dataset):
            question = example['question']
            context = example['context']
            
            # Prédiction
            prediction = self.generate_answer(question, context)
            
            # Ground truth
            if example['answers']['text']:
                ground_truth = example['answers']['text'][0]
                is_answerable = True
            else:
                ground_truth = "[UNANSWERABLE]"
                is_answerable = False
            
            # Métriques
            em = compute_exact_match(prediction, ground_truth)
            f1 = compute_f1(prediction, ground_truth)
            
            # Affichage
            print(f"\n{'─'*80}")
            print(f"EXAMPLE {i+1} {'(Answerable)' if is_answerable else '(Unanswerable)'}")
            print(f"{'─'*80}")
            
            print(f"\n Context (excerpt): {context[:200]}...")
            print(f"\n Question: {question}")
            print(f"\n Prediction: {prediction}")
            print(f" Référence:  {ground_truth}")
            
            print(f"\n Métriques:")
            print(f"   Exact Match: {em} ({'' if em else 'X'})")
            print(f"   F1 Score:    {f1:.2%}")
            
            if i < num_examples - 1:
                input("\n[Appuyez sur Entrée pour continuer...]")

def main():
    """
    Fonction principale.
    """
    import sys
    import os
    
    model_path = "./qa_model_squad"
    
    if not os.path.exists(model_path):
        print("model not found.")
        return
    
    print("EVALUATION MODEL SQUAD")
    
    choice = 1
    
    try:
        evaluator = SQuADEvaluator(model_path)
        
        if choice == 1:
            # Rapide
            evaluator.evaluate(num_examples=100)
        
        elif choice == 2:
                evaluator.evaluate()
           
        else:
            evaluator.evaluate(num_examples=100)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()