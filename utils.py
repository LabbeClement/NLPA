import re
import string

def clean_text(text):
    """
    IMPROVED Text cleaning function that preserves important signals for fake news detection.
    
    What we remove:
    1. URLs (fake news often has suspicious links)
    2. HTML tags
    3. Excessive spaces
    
    What we PRESERVE (important signals):
    1. Case (SHOUTING in all caps is a fake news signal)
    2. Punctuation (!!!, ???, emotional emphasis)
    3. Contractions (won't, don't, can't - natural language)
    4. Special characters (important for tone)
    
    The model will learn that:
    - "BREAKING!!!" is different from "breaking"
    - "You WON'T BELIEVE" is different from "you wont believe"
    - Excessive punctuation signals sensationalism
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Remove URLs (ESSENTIAL for fake news detection)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # 3. Normalize excessive spaces (but keep single spaces)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # That's it! We preserve everything else.
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