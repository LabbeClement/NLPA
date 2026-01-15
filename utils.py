import re
import string

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Normalize excessive spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def analyze_complexity(text):
    """
    Returns technical metrics on the text for the dashboard.
    """
    words = text.split()
    sentences = text.split('.')
    sentences = [s for s in sentences if len(s) > 2]  # Filter empty sentences
    
    word_count = len(words)
    sentence_count = len(sentences)
    avg_sentence_len = word_count / sentence_count if sentence_count > 0 else 0
    
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_len": round(avg_sentence_len, 2)
    }