from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QAEngine:
    def __init__(self):
        # Utilisation de FLAN-T5 pour la comprehension d'instruction
        self.model_name = "google/flan-t5-base"
        print(f"Initialisation du moteur generatif: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def answer_question(self, question, context):
        # Formatage du prompt Seq2Seq
        input_text = f"question: {question} context: {context}"
        
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=1024,
            truncation=True
        ).input_ids.to(self.device)

        # Generation deterministe
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=False,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer