import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from utils import clean_text

def evaluate():
    print("Chargement du modele pour evaluation technique...")
    model_path = "./fake_news_model"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except:
        print("Erreur: Modele non trouve. Lancez train_classifier.py d'abord.")
        return

    # Chargement d'un set de test independant
    dataset = load_dataset("gonzaloA/fake_news", split="test[:100]") 
    
    true_labels = []
    pred_labels = []
    
    print("Inference sur le jeu de test...")
    for item in dataset:
        text = clean_text(item['title'] + " " + item['text'])
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits).item()
            
        true_labels.append(item['label'])
        pred_labels.append(prediction)

    # Matrice de confusion
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'], 
                yticklabels=['Fake', 'Real'])
    plt.ylabel('Verite Terrain')
    plt.xlabel('Prediction Modele')
    plt.title('Matrice de Confusion (Performance Classification)')
    
    filename = 'confusion_matrix.png'
    plt.savefig(filename)
    print(f"Graphique sauvegarde : {filename}")
    
    print("\n--- METRIQUES DETAILLEES ---")
    print(classification_report(true_labels, pred_labels, target_names=['Fake', 'Real']))

if __name__ == "__main__":
    evaluate()