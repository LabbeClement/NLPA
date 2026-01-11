import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from qa_engine import QAEngine
from utils import clean_text, analyze_complexity

# Configuration Page
st.set_page_config(page_title="NLP Analysis Dashboard", layout="wide")

# --- CHARGEMENT MODELES ---
@st.cache_resource
def load_classifier():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./fake_news_model")
        model = AutoModelForSequenceClassification.from_pretrained("./fake_news_model")
        return pipeline("text-classification", model=model, tokenizer=tokenizer)
    except OSError:
        st.error("Erreur critique : Modele de classification introuvable.")
        return None

@st.cache_resource
def load_qa():
    return QAEngine()

# --- INTERFACE ---
st.title("Systeme d'Analyse Automatique de Contenu")
st.markdown("Plateforme de detection de desinformation et d'extraction de connaissances.")

# Layout principal
col_main, col_qa = st.columns([1, 1])

# Initialisation Session
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'classification_label' not in st.session_state: st.session_state.classification_label = ""

with col_main:
    st.subheader("1. Entree Texte")
    input_text = st.text_area("Inserez le contenu a analyser", height=300)
    analyze_btn = st.button("Executer l'analyse")

# SIDEBAR : MONITORING TECHNIQUE
with st.sidebar:
    st.header("Metadonnees Techniques")
    st.markdown("Statistiques du flux entrant")
    
    if input_text:
        # Appel a utils.py
        metrics = analyze_complexity(input_text)
        
        st.metric("Nombre de mots", metrics["word_count"])
        st.metric("Nombre de phrases", metrics["sentence_count"])
        st.metric("Moyenne mots/phrase", metrics["avg_len"])
        
        st.markdown("---")
        st.markdown("**Preprocessing applique :**")
        st.text("Lowercase\nURL removal\nNoise reduction")

# LOGIQUE ANALYSE
if analyze_btn and input_text:
    with st.spinner('Traitement NLP en cours...'):
        # 1. Preprocessing
        clean_input = clean_text(input_text)
        
        # 2. Classification
        classifier = load_classifier()
        if classifier:
            result = classifier(clean_input)[0]
            # Mapping Dataset GonzaloA : 0=Fake, 1=Real
            label = "FAKE" if result['label'] == "LABEL_0" else "REAL"
            score = result['score']
            
            st.session_state.classification_label = label
            st.session_state.analysis_done = True
            
            # Affichage Resultat Strict
            color = "#d9534f" if label == "FAKE" else "#5cb85c"
            bg_color = "#f9f9f9"
            
            st.markdown(f"""
            <div style="
                margin-top: 20px;
                padding: 15px;
                background-color: {bg_color};
                border-left: 5px solid {color};
                border-radius: 2px;">
                <h3 style="color: {color}; margin: 0;">VERDICT : {label}</h3>
                <p style="margin: 5px 0 0 0; color: #333;">Indice de confiance : <strong>{score:.4f}</strong></p>
            </div>
            """, unsafe_allow_html=True)

# LOGIQUE QUESTIONS/REPONSES
with col_qa:
    if st.session_state.analysis_done:
        st.subheader("2. Extraction de Connaissances")
        st.info("Module actif : Google FLAN-T5 (Sequence-to-Sequence)")
        
        user_question = st.text_input("Requete en langage naturel :")
        
        if st.button("Generer la reponse") and user_question:
            with st.spinner('Generation en cours...'):
                qa = load_qa()
                answer = qa.answer_question(user_question, input_text)
                
                st.markdown("**Reponse du systeme :**")
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