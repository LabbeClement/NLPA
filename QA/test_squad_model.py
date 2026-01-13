"""
Interactive test script for the model trained on SQuAD.
Allows asking questions on any text.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os

class SQuADQASystem:
    def __init__(self, model_path="./qa_model_squad"):
        """
        Q&A system based on the SQuAD model.
        """
        print("Q&A SYSTEM - SQUAD MODEL")
        print("="*80)
        
        if not os.path.exists(model_path):
            print(f"\n ERROF: Model not found in {model_path}")
            print("   run : python train_on_squad.py")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"\nLoading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        
        print(f" Model loaded on {self.device}")
    
    def answer_question(self, question, context, show_details=False):
        """
        Answers a question based on context.
        """
        # Prepare input
        input_text = f"question: {question} context: {context}"
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                num_beams=4,
                early_stopping=True
            )
        
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if show_details:
            print(f"\nDetails:")
            print(f"  Input length: {len(inputs['input_ids'][0])} tokens")
            print(f"  Output length: {len(outputs[0])} tokens")
        
        return answer
    
    def interactive_mode(self):
        """
        Interactive mode: ask questions on a context.
        """
        print("INTERACTIVE MODE")
        print("="*80)

        
        while True:
            print("\n" + "-"*80)
            context = input("\n Context: ").strip()
            
            if context.lower() == 'quit':
                print("\nGoodbye!")
                break
            
            if not context:
                print(" Empty context, please try again.")
                continue
            
            print(f"\n Context registered ({len(context)} characters)")
            
            # Loop questions on ce context
            while True:
                question = input("\nQuestion: ").strip()
                
                if question.lower() == 'quit':
                    print("\nGoodbye!")
                    return
                
                if question.lower() == 'new':
                    break
                
                if not question:
                    continue
                
                # Répondre
                answer = self.answer_question(question, context)
                print(f"\n Answer: {answer}")
    
    def demo_examples(self):
        print("DEMONSTRATION - EXAMPLES")
        
        examples = [
            {
                "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower. Constructed from 1887 to 1889, it was initially criticized by some France's leading artists and intellectuals for its design, but it has become a global cultural icon France and one the most recognizable structures in the world. The tower is 330 metres (1,083 ft) tall.",
                "questions": [
                    "Where is the Eiffel Tower located?",
                    "Who designed the Eiffel Tower?",
                    "How tall is the Eiffel Tower?",
                    "When was it built?"
                ]
            },
            {
                "context": "Artificial intelligence was founded as an academic discipline in 1956, and in the years since has experienced several waves optimism, followed by disappointment and the loss funding (known as an 'AI winter'), followed by new approaches, success and renewed funding. AI research has tried and discarded many different approaches during its lifetime, including simulating the brain, modeling human problem solving, formal logic, large databases knowledge and imitating animal behavior.",
                "questions": [
                    "When was AI founded?",
                    "What is an AI winter?",
                    "What approaches has AI research tried?"
                ]
            }
        ]
        
        for i, example in enumerate(examples, 1):
            print(f"\n{'='*80}")
            print(f"EXAMPLE {i}")
            print(f"{'='*80}")
            
            print(f"\n Context:")
            print(f"{example['context']}")
            
            for question in example['questions']:
                answer = self.answer_question(question, example['context'])
                print(f"\n {question}")
                print(f" {answer}")
            
            input(f"\n[Appuyez on Entrée to continuer...]")

def test_with_fake_news():
    """
    Teste the model SQuAD on un article fake news.
    """
    print("TEST ON FAKE NEWS")
    
    try:
        from datasets import load_dataset
        
        qa = SQuADQASystem()
        
        print("\nLoading un article fake news...")
        fake_news = load_dataset("gonzaloA/fake_news", split="test[:1]")
        
        article = fake_news[0]
        context = article['title'] + " " + article['text']
        label = "FAKE" if article['label'] == 0 else "REAL"
        
        print(f"\n Article ({label}):")
        print(f"Title: {article['title']}")
        print(f"Text (excerpt): {context[:300]}...")
        
        # Questions pré-définies
        questions = [
            "What is the main claim?",
            "Who are the sources?",
            "What evidence is provided?",
            "Is this credible?"
        ]
        
        print("\n" + "-"*80)
        print("QUESTIONS & ANSWERS")
        print("-"*80)
        
        for question in questions:
            answer = qa.answer_question(question, context)
            print(f"\n {question}")
            print(f" {answer}")

        
    except ImportError:
        print(" Dataset 'datasets' unavailable")
    except Exception as e:
        print(f" Error: {e}")

def main():

    choice = 1
    try:
        qa = SQuADQASystem()
        
        if choice == 1:
            qa.interactive_mode()
        elif choice == 2:
            qa.demo_examples()
        elif choice == 3:
            test_with_fake_news()
        elif choice == 4:
            qa.demo_examples()
            test_with_fake_news()
            print("\n" + "="*80)
            qa.interactive_mode()
        else:
            print("Invalid choice. Starting demo mode...")
            qa.demo_examples()
    
    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nrun : python train_on_squad.py")
    except Exception as e:
        print(f"\n Error: {e}")

if __name__ == "__main__":
    main()