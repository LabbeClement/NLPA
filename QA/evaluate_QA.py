"""
Evaluation script complet pour the model Q&A.
Métriques: ROUGE, BLEU, BERTScore, Exact Match, F1.
Analyse qualitative: hallucination, pertinence, cohérence.
"""

import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# Métriques
try:
    from rouge_score import rouge_scorer
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from bert_score import score as bert_score
    METRICS_AVAILABLE = True
except ImportError:
    print(" Installation métriques requise:")
    print("  pip install rouge-score nltk bert-score")
    METRICS_AVAILABLE = False

class QAEvaluator:
    def __init__(self, model_path="./qa_model_finetuned", baseline_model="google/flan-t5-base"):
        """
        Évaluateur pour comparer the model fine-tuné vs baseline.
        """
        print("="*80)
        print("INITIALISATION DE L'ÉVALUATEUR Q&A")
        print("="*80)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        
        # Load the model fine-tuné
        print(f"\nLoading the model fine-tuné: {model_path}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
            self.model.eval()
            print("   Model fine-tuné loaded")
        except Exception as e:
            print(f"   Error: {e}")
            raise
        
        # Load the model baseline pour comparaison
        print(f"\nLoading the model baseline: {baseline_model}")
        try:
            self.baseline_tokenizer = AutoTokenizer.from_pretrained(baseline_model)
            self.baseline_model = AutoModelForSeq2SeqLM.from_pretrained(baseline_model).to(self.device)
            self.baseline_model.eval()
            print("   Model baseline loaded")
        except Exception as e:
            print(f"   Baseline non available: {e}")
            self.baseline_model = None
        
        # Initialiser the scorers
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], 
                use_stemmer=True
            )
            self.smoothing = SmoothingFunction().method1
        
        self.results = {
            'finetuned': defaultdict(list),
            'baseline': defaultdict(list)
        }
    
    def load_test_data(self, test_file="qa_dataset_test.json"):
        """
        Charge the dataset of test.
        """
        print(f"\nLoading dataset of test: {test_file}")
        
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print(f"   {len(test_data)} examples of test")
        
        # Statistics
        fake_count = sum(1 for x in test_data if x['label'] == 0)
        real_count = sum(1 for x in test_data if x['label'] == 1)
        
        print(f"\nDistribution:")
        print(f"  - FAKE: {fake_count} ({fake_count/len(test_data)*100:.1f}%)")
        print(f"  - REAL: {real_count} ({real_count/len(test_data)*100:.1f}%)")
        
        # Question types
        q_types = defaultdict(int)
        for item in test_data:
            q_types[item['question_type']] += 1
        
        print(f"\nQuestion types:")
        for q_type, count in sorted(q_types.items()):
            print(f"  - {q_type}: {count}")
        
        return test_data
    
    def generate_answer(self, question, context, model, tokenizer):
        """
        Generates an answer avec un model donné.
        """
        input_text = f"question: {question} context: {context}"
        
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer
    
    def compute_exact_match(self, prediction, reference):
        """
        Exact Match (EM) : prédiction == référence (après normalisation).
        """
        def normalize(text):
            return ' '.join(text.lower().strip().split())
        
        return int(normalize(prediction) == normalize(reference))
    
    def compute_f1(self, prediction, reference):
        """
        F1 token-level entre prédiction et référence.
        """
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        
        if len(pred_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0
        
        common = pred_tokens & ref_tokens
        
        if len(common) == 0:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(ref_tokens)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1
    
    def compute_rouge(self, prediction, reference):
        """
        Calcule the scores ROUGE-1, ROUGE-2, ROUGE-L.
        """
        if not METRICS_AVAILABLE:
            return {}
        
        scores = self.rouge_scorer.score(reference, prediction)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def compute_bleu(self, prediction, reference):
        """
        Calcule the score BLEU.
        """
        if not METRICS_AVAILABLE:
            return 0.0
        
        pred_tokens = prediction.lower().split()
        ref_tokens = [reference.lower().split()]
        
        if len(pred_tokens) == 0:
            return 0.0
        
        score = sentence_bleu(
            ref_tokens, 
            pred_tokens,
            smoothing_function=self.smoothing
        )
        
        return score
    
    def detect_hallucination(self, answer, context):
        """
        Detects if answer contains information not present in context.
        Simple heuristic: searches entities nommées dans answer that are not ins context.
        """
        # Extraire the mots importants of answer (approximation)
        answer_words = set(word.lower() for word in re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', answer))
        context_lower = context.lower()
        
        if len(answer_words) == 0:
            return 0.0  # Pas of mots à vérifier
        
        # Check si the mots sont dans context
        missing = sum(1 for word in answer_words if word.lower() not in context_lower)
        
        hallucination_score = missing / len(answer_words) if answer_words else 0.0
        return hallucination_score
    
    def evaluate_model(self, test_data, model, tokenizer, model_name="model"):
        """
        Évalue un model sur the dataset of test.
        """
        print(f"\n{'='*80}")
        print(f"EVALUATION: {model_name.upper()}")
        print(f"{'='*80}")
        
        all_metrics = {
            'em': [],
            'f1': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'bleu': [],
            'hallucination': []
        }
        
        # Métriques par type of question
        metrics_by_type = defaultdict(lambda: defaultdict(list))
        
        # Métriques par label (fake/real)
        metrics_by_label = defaultdict(lambda: defaultdict(list))
        
        print(f"\nGenerating answers...")
        for item in tqdm(test_data, desc=f"Evaluating {model_name}"):
            question = item['question']
            context = item['context']
            reference = item['answer']
            q_type = item['question_type']
            label = item['label']
            
            # Generate the prédiction
            prediction = self.generate_answer(question, context, model, tokenizer)
            
            # Calculate the métriques
            em = self.compute_exact_match(prediction, reference)
            f1 = self.compute_f1(prediction, reference)
            rouge_scores = self.compute_rouge(prediction, reference)
            bleu = self.compute_bleu(prediction, reference)
            hallucination = self.detect_hallucination(prediction, context)
            
            # Ajouter aux results globaux
            all_metrics['em'].append(em)
            all_metrics['f1'].append(f1)
            all_metrics['bleu'].append(bleu)
            all_metrics['hallucination'].append(hallucination)
            
            if METRICS_AVAILABLE:
                all_metrics['rouge1'].append(rouge_scores.get('rouge1', 0))
                all_metrics['rouge2'].append(rouge_scores.get('rouge2', 0))
                all_metrics['rougeL'].append(rouge_scores.get('rougeL', 0))
            
            # Par type of question
            metrics_by_type[q_type]['em'].append(em)
            metrics_by_type[q_type]['f1'].append(f1)
            metrics_by_type[q_type]['rouge1'].append(rouge_scores.get('rouge1', 0))
            
            # Par label
            label_str = 'fake' if label == 0 else 'real'
            metrics_by_label[label_str]['em'].append(em)
            metrics_by_label[label_str]['f1'].append(f1)
            metrics_by_label[label_str]['rouge1'].append(rouge_scores.get('rouge1', 0))
        
        # Calculate the moyennes
        results = {
            'overall': {k: np.mean(v) for k, v in all_metrics.items()},
            'by_question_type': {
                q_type: {k: np.mean(v) for k, v in metrics.items()}
                for q_type, metrics in metrics_by_type.items()
            },
            'by_label': {
                label: {k: np.mean(v) for k, v in metrics.items()}
                for label, metrics in metrics_by_label.items()
            }
        }
        
        # Display the results
        self._print_results(results, model_name)
        
        return results
    
    def _print_results(self, results, model_name):
        """
        Affiche the results of manière formatée.
        """
        print(f"\n{'─'*80}")
        print(f"RÉSULTATS GLOBAUX - {model_name}")
        print(f"{'─'*80}")
        
        overall = results['overall']
        print(f"\n{'Métrique':<20} {'Score':>10}")
        print(f"{'-'*32}")
        print(f"{'Exact Match':<20} {overall['em']:>10.4f}")
        print(f"{'F1 Token':<20} {overall['f1']:>10.4f}")
        print(f"{'BLEU':<20} {overall['bleu']:>10.4f}")
        if METRICS_AVAILABLE:
            print(f"{'ROUGE-1':<20} {overall['rouge1']:>10.4f}")
            print(f"{'ROUGE-2':<20} {overall['rouge2']:>10.4f}")
            print(f"{'ROUGE-L':<20} {overall['rougeL']:>10.4f}")
        print(f"{'Hallucination':<20} {overall['hallucination']:>10.4f}")
        
        # Par type of question
        print(f"\n{'─'*80}")
        print(f"RÉSULTATS PAR TYPE DE QUESTION")
        print(f"{'─'*80}")
        
        for q_type, metrics in results['by_question_type'].items():
            print(f"\n{q_type.upper()}:")
            print(f"  EM: {metrics['em']:.4f} | F1: {metrics['f1']:.4f} | ROUGE-1: {metrics.get('rouge1', 0):.4f}")
        
        # Par label (fake/real)
        print(f"\n{'─'*80}")
        print(f"RÉSULTATS PAR TYPE D'ARTICLE")
        print(f"{'─'*80}")
        
        for label, metrics in results['by_label'].items():
            print(f"\n{label.upper()}:")
            print(f"  EM: {metrics['em']:.4f} | F1: {metrics['f1']:.4f} | ROUGE-1: {metrics.get('rouge1', 0):.4f}")
    
    def compare_models(self, test_data):
        """
        Compare the model fine-tuné avec the baseline.
        """
        print("\n" + "="*80)
        print("COMPARAISON MODEL FINE-TUNÉ vs BASELINE")
        print("="*80)
        
        # Évaluer fine-tuné
        results_ft = self.evaluate_model(test_data, self.model, self.tokenizer, "Fine-tuned")
        
        # Évaluer baseline
        if self.baseline_model is not None:
            results_bl = self.evaluate_model(
                test_data, 
                self.baseline_model, 
                self.baseline_tokenizer, 
                "Baseline"
            )
            
            # Display the comparaison
            self._print_comparison(results_ft, results_bl)
            
            # Create the graphiques
            self._create_visualizations(results_ft, results_bl)
        
        return results_ft, results_bl if self.baseline_model else None
    
    def _print_comparison(self, results_ft, results_bl):
        """
        Affiche the comparaison entre fine-tuned et baseline.
        """
        print("COMPARAISON DIRECTE")
    
        print(f"\n{'Métrique':<20} {'Fine-tuned':>12} {'Baseline':>12} {'Amélioration':>15}")
        print(f"{'-'*62}")
        
        for metric in ['em', 'f1', 'bleu', 'rouge1', 'rougeL', 'hallucination']:
            if metric not in results_ft['overall']:
                continue
            
            ft_score = results_ft['overall'][metric]
            bl_score = results_bl['overall'][metric]
            
            # Pour hallucination, moins c'est mieux
            if metric == 'hallucination':
                improvement = (bl_score - ft_score) / bl_score * 100 if bl_score > 0 else 0
                improvement_str = f"{improvement:+.1f}%"
            else:
                improvement = (ft_score - bl_score) / bl_score * 100 if bl_score > 0 else 0
                improvement_str = f"{improvement:+.1f}%"
            
            print(f"{metric.upper():<20} {ft_score:>12.4f} {bl_score:>12.4f} {improvement_str:>15}")
    
    def _create_visualizations(self, results_ft, results_bl):
        """
        Crée visualisations of comparaison.
        """
        print("\nCréation visualisations...")
        
        # 1. Comparaison métriques globales
        metrics = ['em', 'f1', 'bleu', 'rouge1', 'rougeL']
        metrics = [m for m in metrics if m in results_ft['overall']]
        
        ft_scores = [results_ft['overall'][m] for m in metrics]
        bl_scores = [results_bl['overall'][m] for m in metrics]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, ft_scores, width, label='Fine-tuned', color='#5cb85c')
        ax.bar(x + width/2, bl_scores, width, label='Baseline', color='#d9534f')
        
        ax.set_ylabel('Score')
        ax.set_title('Comparaison métriques Q&A')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qa_metrics_comparison.png', dpi=300)
        print("   qa_metrics_comparison.png")
        
        # 2. Performance par type of question
        fig, ax = plt.subplots(figsize=(12, 6))
        
        q_types = list(results_ft['by_question_type'].keys())
        ft_f1_by_type = [results_ft['by_question_type'][qt]['f1'] for qt in q_types]
        bl_f1_by_type = [results_bl['by_question_type'][qt]['f1'] for qt in q_types]
        
        x = np.arange(len(q_types))
        
        ax.bar(x - width/2, ft_f1_by_type, width, label='Fine-tuned', color='#5cb85c')
        ax.bar(x + width/2, bl_f1_by_type, width, label='Baseline', color='#d9534f')
        
        ax.set_ylabel('F1 Score')
        ax.set_title('Performance par type of question')
        ax.set_xticks(x)
        ax.set_xticklabels(q_types, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qa_performance_by_type.png', dpi=300)
        print("   qa_performance_by_type.png")
        
        plt.close('all')

def main():
    """
    Fonction principale of evaluation.
    """
    import os
    
    # Check que the model existe
    model_path = "./qa_model_finetuned"
    test_file = "qa_dataset_test.json"
    
    if not os.path.exists(model_path):
        print(" ERREUR: Model fine-tuné introuvable!")
        print(f"   Exécutez of abord: python train_qa_model.py")
        return
    
    if not os.path.exists(test_file):
        print(" ERREUR: Dataset of test introuvable!")
        print(f"   Exécutez of abord: python generate_qa_dataset.py")
        return
    
    # Create l'évaluateur
    evaluator = QAEvaluator(model_path=model_path)
    
    # Load the data of test
    test_data = evaluator.load_test_data(test_file)
    
    # Comparer the models
    results_ft, results_bl = evaluator.compare_models(test_data)
    
    # Save the results
    results = {
        'finetuned': results_ft,
        'baseline': results_bl
    }
    
    with open('qa_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
if __name__ == "__main__":
    main()