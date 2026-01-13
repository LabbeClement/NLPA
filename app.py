import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from qa_engine import QAEngine
from utils import clean_text, analyze_complexity

# Configuration Page
st.set_page_config(page_title="NLP Analysis Dashboard", layout="wide")

# Load model
@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./fake_news_model")
        model = AutoModelForSequenceClassification.from_pretrained("./fake_news_model")
        return model, tokenizer
    except OSError:
        st.error("Critical error: Classification model not found.")
        return None, None

@st.cache_resource
def load_qa():
    return QAEngine()

def get_word_attributions(text, model, tokenizer):
    """
    Extract word importance using gradient-based attribution.
    Shows which words the model actually used for its decision.
    """
    try:
        # Set model to eval mode but enable gradients
        model.eval()
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get embeddings with gradient tracking
        embedding_layer = model.get_input_embeddings()
        
        # Create embeddings that require grad
        input_ids = inputs['input_ids']
        original_embeddings = embedding_layer(input_ids)
        original_embeddings = original_embeddings.detach().requires_grad_(True)
        
        # Forward pass with custom embeddings
        outputs = model(
            inputs_embeds=original_embeddings, 
            attention_mask=inputs['attention_mask'],
            return_dict=True
        )
        logits = outputs.logits
        
        # Get prediction
        predicted_class = torch.argmax(logits, dim=1).item()
        
        # Clear any existing gradients
        if original_embeddings.grad is not None:
            original_embeddings.grad.zero_()
        
        # Backward pass to get gradients
        score = logits[0, predicted_class]
        score.backward(retain_graph=False)
        
        # Check if gradients were computed
        if original_embeddings.grad is None:
            # Fallback: use attention weights if gradients fail
            return get_attention_weights(text, model, tokenizer)
        
        # Get gradient magnitudes
        gradients = original_embeddings.grad
        attributions = (gradients * original_embeddings).sum(dim=-1).squeeze().abs().detach().numpy()
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        # Combine tokens and attributions
        token_attributions = []
        for token, attr in zip(tokens, attributions):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_attributions.append((token, float(attr)))
        
        # Sort by importance
        token_attributions.sort(key=lambda x: x[1], reverse=True)
        
        return token_attributions
        
    except Exception as e:
        st.warning(f"Gradient computation failed: {e}. Using attention weights fallback.")
        return get_attention_weights(text, model, tokenizer)

def get_attention_weights(text, model, tokenizer):
    """
    Fallback method using attention weights when gradients fail.
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model outputs with attention
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # Average attention weights across all heads and layers
    attentions = outputs.attentions  # Tuple of attention tensors
    
    # Sum attention across all layers and heads
    # Shape: (layers, batch, heads, seq_len, seq_len)
    all_attention = torch.stack([attn.squeeze(0).mean(dim=0) for attn in attentions])
    
    # Average across layers, then sum across query positions (columns)
    avg_attention = all_attention.mean(dim=0).sum(dim=0)  # Sum over queries
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Combine tokens and attention scores
    token_attributions = []
    for token, score in zip(tokens, avg_attention):
        if token not in ['[CLS]', '[SEP]', '[PAD]']:
            token_attributions.append((token, float(score)))
    
    # Sort by importance
    token_attributions.sort(key=lambda x: x[1], reverse=True)
    
    return token_attributions

def highlight_important_words(text, token_attributions, top_n=10):
    """
    Generate HTML with highlighted important words based on model's decision.
    """
    # Get top N most important tokens
    top_tokens = dict(token_attributions[:top_n])
    
    # Normalize attribution scores for coloring
    if top_tokens:
        max_attr = max(top_tokens.values())
        if max_attr > 0:
            top_tokens = {k: v/max_attr for k, v in top_tokens.items()}
    
    # Split text into words and highlight
    words = text.split()
    highlighted_words = []
    
    for word in words:
        word_lower = word
        # Remove punctuation for matching
        clean_word = word_lower.strip('.,!?;:')
        
        # Check if word or part of word is in important tokens
        importance = 0
        for token, score in top_tokens.items():
            token_clean = token.replace('##', '')
            if token_clean in clean_word or clean_word in token_clean:
                importance = max(importance, score)
        
        if importance > 0:
            # Color intensity based on importance
            intensity = int(255 - (importance * 200))  # From red (high) to light (low)
            color = f"rgb(255, {intensity}, {intensity})"
            highlighted_words.append(f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{word}</span>')
        else:
            highlighted_words.append(word)
    
    return ' '.join(highlighted_words)

# --- INTERFACE ---
st.title("Automated Content Analysis System")
st.markdown("Platform for misinformation detection and knowledge extraction.")

# Main layout
col_main, col_qa = st.columns([1, 1])

# Session initialization
if 'analysis_done' not in st.session_state: 
    st.session_state.analysis_done = False
if 'classification_label' not in st.session_state: 
    st.session_state.classification_label = ""
if 'token_attributions' not in st.session_state:
    st.session_state.token_attributions = None
if 'original_text' not in st.session_state:
    st.session_state.original_text = ""

with col_main:
    st.subheader("1. Text Input")
    input_text = st.text_area("Insert content to analyze", height=300)
    analyze_btn = st.button("Execute Analysis")

# SIDEBAR: TECHNICAL MONITORING
with st.sidebar:
    st.header("Technical Metadata")
    st.markdown("Input stream statistics")
    
    if input_text:
        metrics = analyze_complexity(input_text)
        
        st.metric("Word count", metrics["word_count"])
        st.metric("Sentence count", metrics["sentence_count"])
        st.metric("Avg words/sentence", metrics["avg_len"])
        
        st.markdown("---")
        st.markdown("**Preprocessing applied:**")
        st.text("Lowercase\nURL removal\nNoise reduction")

# ANALYSIS LOGIC
if analyze_btn and input_text:
    with st.spinner('NLP processing in progress...'):
        # Load model
        model, tokenizer = load_model_and_tokenizer()
        
        if model is not None and tokenizer is not None:
            # 1. Preprocessing
            clean_input = clean_text(input_text)
            
            # 2. Classification with attention
            inputs = tokenizer(clean_input, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
            
            probabilities = torch.softmax(logits, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()
            
            # Label assignment - if your model predicts backwards, flip this line:
            # Current: 0=FAKE, 1=REAL (standard for gonzaloA/fake_news dataset)
            # If results are inverted, change to: label = "REAL" if predicted_class == 0 else "FAKE"
            label = "FAKE" if predicted_class == 0 else "REAL"
            score = probabilities[predicted_class].item()
            
            # 3. Get word attributions (model-based explainability)
            token_attributions = get_word_attributions(clean_input, model, tokenizer)
            
            st.session_state.classification_label = label
            st.session_state.analysis_done = True
            st.session_state.token_attributions = token_attributions
            st.session_state.original_text = input_text
            
            # Main result display
            color = "#d9534f" if label == "FAKE" else "#5cb85c"
            bg_color = "#f9f9f9"
            
            st.markdown(f"""
            <div style="
                margin-top: 20px;
                padding: 15px;
                background-color: {bg_color};
                border-left: 5px solid {color};
                border-radius: 2px;">
                <h3 style="color: {color}; margin: 0;">VERDICT: {label}</h3>
                <p style="margin: 5px 0 0 0; color: #333;">Confidence score: <strong>{score:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # MODEL-BASED EXPLAINABILITY
            st.markdown("---")
            st.subheader("Model Decision Explanation")
            
            
            # Show highlighted text
            st.markdown("### Important words identified by the model:")
            highlighted_html = highlight_important_words(input_text, token_attributions)
            st.markdown(f'<div style="padding: 15px; background-color: #f9f9f9; border-radius: 5px; line-height: 1.8;">{highlighted_html}</div>', unsafe_allow_html=True)
            
            # Show top influencing words with scores
            st.markdown("### Top 15 most influential tokens:")
            
            col1, col2, col3 = st.columns(3)
            top_15 = token_attributions[:15]
            
            for i, (token, score) in enumerate(top_15):
                col = [col1, col2, col3][i % 3]
                with col:
                    # Clean token display (remove ## from wordpiece tokens)
                    display_token = token.replace('##', '')
                    st.metric(display_token, f"{score:.4f}")
# Q&A LOGIC
with col_qa:
    if st.session_state.analysis_done:

        user_question = st.text_input("Natural language query:")
        
        if st.button("Generate answer") and user_question:
            with st.spinner('Generation in progress...'):
                qa = load_qa()
                answer = qa.answer_question(user_question, input_text)
                
                st.markdown("**System response:**")
                st.markdown(f"""
                <div style="
                    background-color: #eef; 
                    padding: 15px; 
                    border: 1px solid #ccd;
                    border-radius: 4px;
                    color: #000;">
                    {answer}
                </div>
                """, unsafe_allow_html=True)