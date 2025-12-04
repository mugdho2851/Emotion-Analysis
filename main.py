import streamlit as st
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
import re
import matplotlib.pyplot as plt
import pandas as pd
import random

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# CUSTOM CSS FOR CLEAN DESIGN
# ======================
st.markdown("""
<style>
    /* Clean light theme */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Headers */
    .clean-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1e293b;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .clean-subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Clean cards */
    .clean-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
    }
    
    .clean-card:hover {
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
        transform: translateY(-2px);
    }
    
    /* Sentiment colors */
    .clean-positive {
        color: #10b981;
        font-weight: 600;
    }
    
    .clean-negative {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Progress bars */
    .clean-progress {
        background: #f1f5f9;
        border-radius: 8px;
        height: 8px;
        margin: 12px 0;
        overflow: hidden;
    }
    
    .clean-progress-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2563eb;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
        font-size: 16px;
        padding: 15px;
        background: white;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }
    
    /* Score display */
    .clean-score {
        background: #f8fafc;
        border-radius: 8px;
        padding: 10px;
        font-family: 'Monaco', 'Courier New', monospace;
        font-size: 14px;
        color: #475569;
        text-align: center;
        margin-top: 10px;
        border: 1px solid #e2e8f0;
    }
    
    /* Model badges */
    .model-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        margin: 0 4px 8px 0;
    }
    
    .rnn-badge {
        background-color: #dbeafe;
        color: #1d4ed8;
    }
    
    .lstm-badge {
        background-color: #dcfce7;
        color: #15803d;
    }
    
    .gru-badge {
        background-color: #f3e8ff;
        color: #7c3aed;
    }
    
    /* Divider */
    .clean-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# INITIALIZATION
# ======================

# Initialize session state
if 'input_text' not in st.session_state:
    st.session_state.input_text = "I love this amazing product!"

# ======================
# SIDEBAR
# ======================

with st.sidebar:
    # Title
    st.markdown("## ⚙️ Settings")
    st.markdown("---")
    
    # Model selection
    st.markdown("### Model Selection")
    col1, col2, col3 = st.columns(3)
    with col1:
        show_rnn = st.checkbox("RNN", value=True, key="rnn")
    with col2:
        show_lstm = st.checkbox("LSTM", value=True, key="lstm")
    with col3:
        show_gru = st.checkbox("GRU", value=True, key="gru")
    
    st.markdown("---")
    
    # Examples
    st.markdown("### Try Examples")
    examples = [
        "I love this amazing product!",
        "Terrible service, very disappointed.",
        "Excellent quality, highly recommend.",
        "Not worth the money at all.",
        "Absolutely fantastic experience!"
    ]
    
    for example in examples:
        if st.button(f"• {example[:25]}...", key=f"ex_{example[:10]}"):
            st.session_state.input_text = example
            st.rerun()
    
    st.markdown("---")
    
    # Info
    st.markdown("### About")
    with st.expander("Project Details"):
        st.write("""
        **Neural Network Models:**
        - RNN (Recurrent Neural Network)
        - LSTM (Long Short-Term Memory)
        - GRU (Gated Recurrent Unit)
        
        **Dataset:** Sentiment140
        **Task:** Binary sentiment classification
        """)

# ======================
# MAIN CONTENT
# ======================

# Header
st.markdown('<h1 class="clean-title">Sentiment Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="clean-subtitle">Analyze text sentiment using neural networks</p>', unsafe_allow_html=True)

# Input Section
st.markdown("### Enter Text")
input_text = st.text_area(
    "Enter your text here:",
    value=st.session_state.input_text,
    height=120,
    placeholder="Type or paste text here...",
    label_visibility="visible"
)

# Action buttons
col1, col2, col3 = st.columns(3)
with col1:
    analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")
with col2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.input_text = ""
        st.rerun()
with col3:
    if st.button("🎲 Random", use_container_width=True):
        random_examples = [
            "The sunset was absolutely breathtaking!",
            "Customer service was extremely helpful.",
            "Product arrived damaged and late.",
            "Exceeded all my expectations!",
            "Would not recommend to anyone."
        ]
        st.session_state.input_text = random.choice(random_examples)
        st.rerun()

# ======================
# MODEL LOADING
# ======================

@st.cache_resource
def load_models():
    """Load all trained models"""
    models_dir = "saved_models"
    models = {}
    
    try:
        # Load tokenizer
        tokenizer_path = os.path.join(models_dir, 'tokenizer.pickle')
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        
        # Load parameters
        params_path = os.path.join(models_dir, 'params.json')
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        # Load models
        model_files = {
            'RNN': 'rnn_model.h5',
            'LSTM': 'lstm_model.h5',
            'GRU': 'gru_model.h5'
        }
        
        for name, filename in model_files.items():
            model_path = os.path.join(models_dir, filename)
            if os.path.exists(model_path):
                models[name] = {
                    'model': keras.models.load_model(model_path),
                    'loaded': True
                }
            else:
                models[name] = {'loaded': False}
        
        return models, tokenizer, params
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# ======================
# PREPROCESSING (FIXED REGEX)
# ======================

def preprocessing(text):
    """Preprocessing function - fixed regex"""
    # Fixed regex pattern (use raw string)
    text_cleaning_re = r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+'
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        tokens.append(token)
    return ' '.join(tokens)

# ======================
# PREDICTION FUNCTION
# ======================

def predict_sentiment(text, models_dict, tokenizer, max_seq_length):
    """Predict sentiment using selected models"""
    # Preprocess
    cleaned_text = preprocessing(text)
    
    # Tokenize
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=max_seq_length,
        padding='post'
    )
    
    predictions = {}
    
    for model_name, model_info in models_dict.items():
        if model_info['loaded']:
            try:
                prediction = model_info['model'].predict(padded, verbose=0)[0][0]
                sentiment = "positive" if prediction > 0.5 else "negative"
                confidence = prediction if sentiment == "positive" else 1 - prediction
                
                predictions[model_name] = {
                    'sentiment': sentiment,
                    'score': float(prediction),
                    'confidence': float(confidence),
                    'loaded': True,
                    'color': '#10b981' if sentiment == 'positive' else '#ef4444'
                }
            except Exception as e:
                predictions[model_name] = {
                    'sentiment': 'error',
                    'score': 0.0,
                    'confidence': 0.0,
                    'loaded': False,
                    'color': '#64748b',
                    'error': str(e)
                }
    
    return predictions

# ======================
# DISPLAY MODEL CARD
# ======================

def display_clean_card(model_name, prediction):
    """Display a clean model card"""
    if prediction['sentiment'] == 'positive':
        sentiment_class = "clean-positive"
        emoji = "😊"
    elif prediction['sentiment'] == 'negative':
        sentiment_class = "clean-negative"
        emoji = "😞"
    else:
        sentiment_class = ""
        emoji = "❓"
    
    # Badge color based on model
    badge_class = {
        'RNN': 'rnn-badge',
        'LSTM': 'lstm-badge',
        'GRU': 'gru-badge'
    }.get(model_name, '')
    
    return f"""
    <div class="clean-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
            <div>
                <h3 style="margin: 0; color: #1e293b;">{model_name}</h3>
                <span class="model-badge {badge_class}">Neural Network</span>
            </div>
            <div style="font-size: 2rem;">{emoji}</div>
        </div>
        
        <div style="margin: 1rem 0;">
            <div style="font-size: 1.5rem; font-weight: 600; margin-bottom: 0.5rem;" class="{sentiment_class}">
                {prediction['sentiment'].upper()}
            </div>
            <div style="color: #64748b; font-size: 0.9rem;">
                Confidence: {prediction['confidence']:.1%}
            </div>
        </div>
        
        <div class="clean-progress">
            <div class="clean-progress-fill" style="width: {prediction['confidence']*100}%; background: {prediction['color']};"></div>
        </div>
        
        <div class="clean-score">
            Score: {prediction['score']:.8f}
        </div>
    </div>
    """

# ======================
# MAIN ANALYSIS LOGIC
# ======================

if analyze_btn and input_text.strip():
    with st.spinner("Analyzing..."):
        # Load models
        models_dict, tokenizer, params = load_models()
        
        if models_dict and tokenizer and params:
            # Get predictions
            predictions = predict_sentiment(
                input_text, 
                models_dict, 
                tokenizer, 
                params.get('MAX_SEQ_LENGTH', 30)
            )
            
            # Filter models based on selection
            filtered_models = []
            if show_rnn and 'RNN' in predictions and predictions['RNN']['loaded']:
                filtered_models.append('RNN')
            if show_lstm and 'LSTM' in predictions and predictions['LSTM']['loaded']:
                filtered_models.append('LSTM')
            if show_gru and 'GRU' in predictions and predictions['GRU']['loaded']:
                filtered_models.append('GRU')
            
            if not filtered_models:
                st.warning("Please select at least one model to analyze.")
            else:
                # Results header
                st.markdown('<div class="clean-divider"></div>', unsafe_allow_html=True)
                st.markdown("## 📊 Analysis Results")
                
                # Display model cards
                cols = st.columns(len(filtered_models))
                for idx, model_name in enumerate(filtered_models):
                    with cols[idx]:
                        pred = predictions[model_name]
                        st.markdown(display_clean_card(model_name, pred), unsafe_allow_html=True)
                
                # Overall sentiment
                st.markdown('<div class="clean-divider"></div>', unsafe_allow_html=True)
                st.markdown("## 🎯 Overall Sentiment")
                
                positive_count = sum(1 for m in filtered_models 
                                   if predictions[m]['sentiment'] == 'positive')
                total = len(filtered_models)
                
                if positive_count > total / 2:
                    overall = "Positive"
                    overall_color = "#10b981"
                    overall_emoji = "😊"
                elif positive_count < total / 2:
                    overall = "Negative"
                    overall_color = "#ef4444"
                    overall_emoji = "😞"
                else:
                    overall = "Neutral"
                    overall_color = "#64748b"
                    overall_emoji = "😐"
                
                # Display overall sentiment
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 2rem; background: white; border-radius: 12px; 
                                border: 1px solid #e2e8f0; margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                        <div style="font-size: 3rem; margin-bottom: 1rem;">{overall_emoji}</div>
                        <div style="font-size: 1.8rem; color: {overall_color}; font-weight: 600; margin-bottom: 0.5rem;">
                            {overall}
                        </div>
                        <div style="color: #64748b;">
                            {positive_count} of {total} models predict positive
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model comparison chart
                if len(filtered_models) > 1:
                    st.markdown('<div class="clean-divider"></div>', unsafe_allow_html=True)
                    st.markdown("## 📈 Model Comparison")
                    
                    # Create simple bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    models_list = filtered_models
                    scores = [predictions[m]['score'] for m in models_list]
                    colors = [predictions[m]['color'] for m in models_list]
                    
                    bars = ax.bar(models_list, scores, color=colors, alpha=0.8, width=0.6)
                    ax.axhline(y=0.5, color='#94a3b8', linestyle='--', alpha=0.5)
                    
                    # Add score labels
                    for bar, score in zip(bars, scores):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{score:.3f}', ha='center', va='bottom', fontsize=10)
                    
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Score')
                    ax.set_title('Prediction Scores by Model')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.grid(True, alpha=0.1, axis='y')
                    
                    st.pyplot(fig)
        
        else:
            st.error("""
            Models not found. Please ensure:
            1. All training notebooks have been run
            2. Model files exist in 'saved_models' folder
            3. Required files: rnn_model.h5, lstm_model.h5, gru_model.h5
            """)

elif analyze_btn and not input_text.strip():
    st.warning("Please enter some text to analyze.")

# ======================
# FOOTER
# ======================

st.markdown('<div class="clean-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem 0; font-size: 0.9rem;">
    <div>Sentiment Analysis Project • Neural Network Models</div>
    <div style="margin-top: 0.5rem; display: flex; justify-content: center; gap: 1.5rem;">
        <span>RNN</span>
        <span>LSTM</span>
        <span>GRU</span>
    </div>
</div>
""", unsafe_allow_html=True)