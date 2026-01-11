import re
import string

def clean_text(text):
    """
    Fonction de nettoyage (Preprocessing) pour normaliser les entrées
    avant l'envoi aux modèles NLP.
    
    Étapes :
    1. Conversion en minuscules
    2. Suppression des URLs
    3. Suppression des balises HTML
    4. Suppression de la ponctuation excessive
    5. Normalisation des espaces
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Minuscules
    text = text.lower()
    
    # 2. Suppression des URLs
    #text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Suppression HTML
    text = re.sub(r'<.*?>', '', text)
    
    # 4. Nettoyage ponctuation ex:  (on garde . , ? pour le sens)
    # On enlève tout ce qui n'est pas alphanumérique ou ponctuation basique
    text = re.sub(r'[^\w\s.,?!]', '', text)
    
    # 5. Espaces multiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def analyze_complexity(text):
    """
    Retourne des métriques techniques sur le texte pour le dashboard.
    """
    words = text.split()
    sentences = text.split('.')
    sentences = [s for s in sentences if len(s) > 2] # Filtrer les phrases vides
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_len": round(avg_sentence_len, 2)
    }