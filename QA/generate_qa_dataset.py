"""
Script to generate a synthetic Q&A dataset for training.
Uses an LLM to create question-answer pairs from articles.
"""

import json
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm import tqdm
from utils import clean_text

class QADatasetGenerator:
    def __init__(self):
        """
        Q&A dataset generator using FLAN-T5-base to create
        high-quality question-answer pairs.
        """
        print("Initializing Q&A generator...")
        self.model_name = "google/flan-t5-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
        
        # Question types to generate
        self.question_types = [
            "factual",      # Factual questions (who, what, when, where)
            "analytical",   # Analytical questions (why, how)
            "verification", # Verification questions (true/false, is it)
            "summarization",# Summary questions (summarize, explain)
            "comparison",   # Comparison questions
            "opinion"       # Opinion questions (according to article)
        ]
    
    def generate_questions(self, text, num_questions=5):
        """
        Generates multiple questions of différents types à partir of un texte.
        """
        questions = []
        
        # Limit text size to avoid overflows
        text_snippet = text[:1500]
        
        prompts = {
            "factual": f"Generate a factual question that can be answered from this text: {text_snippet}\nQuestion:",
            "analytical": f"Generate an analytical question asking 'why' or 'how' about this text: {text_snippet}\nQuestion:",
            "verification": f"Generate a yes/no verification question about this text: {text_snippet}\nQuestion:",
            "summarization": f"Generate a question asking for a summary or explanation from this text: {text_snippet}\nQuestion:",
            "comparison": f"Generate a question asking to compare or contrast elements in this text: {text_snippet}\nQuestion:",
        }
        
        # Generate questions of different types
        selected_types = random.sample(list(prompts.keys()), min(num_questions, len(prompts)))
        
        for q_type in selected_types:
            try:
                input_ids = self.tokenizer(
                    prompts[q_type],
                    return_tensors="pt",
                    max_length=512,
                    truncation=True
                ).input_ids.to(self.device)
                
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    num_return_sequences=1
                )
                
                question = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                questions.append((question, q_type))
                
            except Exception as e:
                print(f"Error generating question {q_type}: {e}")
                continue
        
        return questions
    
    def generate_answer(self, question, context):
        """
        Generates an answer from context.
        """
        prompt = f"Answer this question based on context.\nContext: {context[:1000]}\nQuestion: {question}\nAnswer:"
        
        try:
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).input_ids.to(self.device)
            
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=False,
                num_beams=4
            )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return ""
    
    def create_qa_dataset(self, num_articles=500, questions_per_article=3, output_file="qa_dataset.json"):
        """
        Creates a complete dataset of Q&A from articles fake/real.
        
        Format dataset:
        {
            "context": "Article text",
            "question": "Question",
            "answer": "Answer",
            "question_type": "factual/analytical/etc",
            "label": 0/1 (fake/real),
            "id": unique_id
        }
        """
        print(f"\nGenerating Q&A dataset...")
        print(f"  - Target articles: {num_articles}")
        print(f"  - Questions per article: {questions_per_article}")
        
        # Load articles
        print("\nLoading articles...")
        try:
            dataset = load_dataset("gonzaloA/fake_news", split='train')
            
            # Balance fake/real
            fake_articles = dataset.filter(lambda x: x['label'] == 0)
            real_articles = dataset.filter(lambda x: x['label'] == 1)
            
            num_per_class = num_articles // 2
            fake_sample = list(fake_articles.select(range(min(num_per_class, len(fake_articles)))))
            real_sample = list(real_articles.select(range(min(num_per_class, len(real_articles)))))
            
            articles = fake_sample + real_sample
            random.shuffle(articles)
            
            print(f"   {len(articles)} articles loaded")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return
        
        # Generate the Q&A
        qa_data = []
        qa_id = 0
        
        print("\nGenerating Q&A...")
        for idx, article in enumerate(tqdm(articles)):
            # Prepare context
            title = str(article.get('title', ''))
            text = str(article.get('text', ''))
            context = clean_text(title + " " + text)
            
            if len(context) < 100:
                continue
            
            label = article['label']
            
            # Generate questions
            questions = self.generate_questions(context, num_questions=questions_per_article)
            
            # Generate answers
            for question, q_type in questions:
                if not question or len(question) < 10:
                    continue
                
                answer = self.generate_answer(question, context)
                
                if not answer or len(answer) < 5:
                    continue
                
                qa_entry = {
                    "id": qa_id,
                    "context": context,
                    "question": question,
                    "answer": answer,
                    "question_type": q_type,
                    "label": label,
                    "article_id": idx
                }
                
                qa_data.append(qa_entry)
                qa_id += 1
        
        # Save
        print(f"\n {len(qa_data)} Q&A pairs generated")
        
        # Statistics
        fake_qa = sum(1 for x in qa_data if x['label'] == 0)
        real_qa = sum(1 for x in qa_data if x['label'] == 1)
        
        print(f"\nStatistics:")
        print(f"  - FAKE articles: {fake_qa} Q&A")
        print(f"  - REAL articles: {real_qa} Q&A")
        
        q_types = {}
        for entry in qa_data:
            q_type = entry['question_type']
            q_types[q_type] = q_types.get(q_type, 0) + 1
        
        print(f"\nQuestion types:")
        for q_type, count in sorted(q_types.items()):
            print(f"  - {q_type}: {count}")
        
        # Save en JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(qa_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n Dataset saved: {output_file}")
        
        # Create train/val/test splits
        random.shuffle(qa_data)
        
        train_split = int(len(qa_data) * 0.8)
        val_split = int(len(qa_data) * 0.9)
        
        splits = {
            'train': qa_data[:train_split],
            'validation': qa_data[train_split:val_split],
            'test': qa_data[val_split:]
        }
        
        for split_name, split_data in splits.items():
            split_file = output_file.replace('.json', f'_{split_name}.json')
            with open(split_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, indent=2, ensure_ascii=False)
            print(f"   {split_name}: {len(split_data)} examples → {split_file}")
        
        return qa_data

def main():
    """
    Fonction principale pour générer the dataset Q&A.
    """
    generator = QADatasetGenerator()
    
    # Generate the dataset
    qa_data = generator.create_qa_dataset(
        num_articles=100,        
        questions_per_article=4,
        output_file="qa_dataset.json"
    )
    
    # Display quelques examples
    if qa_data:
        print("\n" + "="*80)
        print("EXAMPLES DE Q&A GENERATED")
        print("="*80)
        
        for i in range(min(3, len(qa_data))):
            example = qa_data[i]
            print(f"\n[Example {i+1}] Type: {example['question_type']} | Label: {'FAKE' if example['label']==0 else 'REAL'}")
            print(f"Context (excerpt): {example['context'][:200]}...")
            print(f"Question: {example['question']}")
            print(f"Answer: {example['answer']}")
            print("-" * 80)

if __name__ == "__main__":
    main()