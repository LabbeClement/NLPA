from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class QAEngine:
    def __init__(self):
        # Using FLAN-T5 for instruction understanding
        self.model_name = "./qa_model_squad"
        print(f"Initializing generative engine: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def answer_question(self, question, context):
        # Seq2Seq prompt formatting
        input_text = f"question: {question} context: {context}"
        
        input_ids = self.tokenizer(
            input_text, 
            return_tensors="pt", 
            max_length=1024,
            truncation=True
        ).input_ids.to(self.device)

        # Deterministic generation
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=150,
            do_sample=False,
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer