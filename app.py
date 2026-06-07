"""
app.py
Fake Review Forensic Analyzer — Streamlit Dashboard
Supports both DistilBERT (deep learning) and TF-IDF (fallback) models.
Run: streamlit run app.py
"""

import os
import joblib
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import torch

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Review Forensics AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Dark Cyberpunk Theme ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: #0a0e1a;
      color: #c9d1d9;
  }
  .stApp {
      background: linear-gradient(135deg, #0a0e1a 0%, #0d1117 50%, #0a0e1a 100%);
  }
  [data-testid="stSidebar"] {
      background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
      border-right: 1px solid #21262d;
  }
  .header-card {
      background: linear-gradient(135deg, #1a1f2e 0%, #161b22 100%);
      border: 1px solid #30363d;
      border-radius: 16px;
      padding: 2rem 2.5rem;
      margin-bottom: 2rem;
      box-shadow: 0 4px 24px rgba(0,0,0,0.4);
  }
  .header-title {
      font-size: 2.2rem;
      font-weight: 700;
      background: linear-gradient(90deg, #58a6ff, #bc8cff, #f778ba);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin: 0;
  }
  .header-subtitle { font-size: 0.95rem; color: #8b949e; margin-top: 0.4rem; }

  .model-badge-bert {
      display: inline-block;
      background: linear-gradient(135deg, #1f3a5f, #0d2137);
      border: 1px solid #58a6ff;
      border-radius: 20px;
      padding: 0.3rem 1rem;
      font-size: 0.8rem;
      color: #58a6ff;
      font-weight: 600;
      margin-top: 0.8rem;
  }
  .model-badge-sklearn {
      display: inline-block;
      background: linear-gradient(135deg, #2d2208, #1a1400);
      border: 1px solid #d29922;
      border-radius: 20px;
      padding: 0.3rem 1rem;
      font-size: 0.8rem;
      color: #d29922;
      font-weight: 600;
      margin-top: 0.8rem;
  }
  .result-fake {
      background: linear-gradient(135deg, #2d1117 0%, #1a0a0a 100%);
      border: 1px solid #f85149;
      border-left: 4px solid #f85149;
      border-radius: 12px;
      padding: 1.5rem 2rem;
      margin: 1rem 0;
      box-shadow: 0 0 20px rgba(248,81,73,0.15);
  }
  .result-authentic {
      background: linear-gradient(135deg, #0d2818 0%, #0a1a0d 100%);
      border: 1px solid #3fb950;
      border-left: 4px solid #3fb950;
      border-radius: 12px;
      padding: 1.5rem 2rem;
      margin: 1rem 0;
      box-shadow: 0 0 20px rgba(63,185,80,0.15);
  }
  .verdict-text { font-size: 1.6rem; font-weight: 700; margin: 0; }
  .verdict-fake { color: #f85149; }
  .verdict-authentic { color: #3fb950; }
  .verdict-sub { font-size: 0.9rem; color: #8b949e; margin-top: 0.3rem; }

  .metric-box {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 10px;
      padding: 1rem 1.2rem;
      text-align: center;
      margin-bottom: 0.8rem;
  }
  .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-size: 1.5rem; font-weight: 700; color: #58a6ff; font-family: 'JetBrains Mono', monospace; }

  .explain-panel {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 1.5rem;
      margin-top: 1rem;
  }
  .explain-title {
      font-size: 0.85rem; color: #8b949e;
      text-transform: uppercase; letter-spacing: 0.08em;
      margin-bottom: 1rem; font-weight: 600;
  }
  .word-fake      { background: rgba(248,81,73,0.25); color: #f85149; border-radius: 4px; padding: 0 4px; font-weight: 600; }
  .word-authentic { background: rgba(63,185,80,0.20); color: #3fb950; border-radius: 4px; padding: 0 4px; font-weight: 600; }
  .word-neutral   { color: #c9d1d9; }
  .highlighted-text { font-size: 1rem; line-height: 2; }

  textarea {
      background-color: #161b22 !important;
      color: #c9d1d9 !important;
      border: 1px solid #30363d !important;
      border-radius: 8px !important;
  }
  .stButton > button {
      background: linear-gradient(135deg, #1f6feb, #388bfd);
      color: white; border: none; border-radius: 8px;
      padding: 0.6rem 2rem; font-weight: 600; font-size: 1rem;
      transition: all 0.2s ease; width: 100%;
  }
  .stButton > button:hover {
      background: linear-gradient(135deg, #388bfd, #58a6ff);
      transform: translateY(-1px);
      box-shadow: 0 4px 12px rgba(56,139,253,0.4);
  }
  hr { border-color: #21262d; }
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0d1117; }
  ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Load Model (BERT preferred, sklearn fallback) ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    bert_dir = "bert_model"

    # ── Try BERT first ─────────────────────────────────────────────────────
    if os.path.exists(bert_dir):
        try:
            from transformers import (
                DistilBertTokenizerFast,
                DistilBertForSequenceClassification,
            )
            tokenizer = DistilBertTokenizerFast.from_pretrained(bert_dir)
            model     = DistilBertForSequenceClassification.from_pretrained(bert_dir)
            model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            meta = joblib.load("model_meta.pkl") if os.path.exists("model_meta.pkl") else {}
            return "bert", model, tokenizer, device, meta
        except Exception as e:
            pass  # Fall through to sklearn

    # ── Fallback: sklearn pipeline ─────────────────────────────────────────
    if os.path.exists("review_model.pkl"):
        pipeline = joblib.load("review_model.pkl")
        meta     = joblib.load("model_meta.pkl") if os.path.exists("model_meta.pkl") else {}
        return "sklearn", pipeline, None, None, meta

    return None, None, None, None, {}


model_type, model, tokenizer, device, meta = load_model()


# ── Prediction Functions ───────────────────────────────────────────────────────
def predict_bert(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        max_length=256, truncation=True, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).cpu().detach().numpy()[0]
    # probs[0] = class 0 = OR (Authentic), probs[1] = class 1 = CG (Fake)
    # If model predicts everything as authentic, swap the labels
    fake_prob = probs[1]
    auth_prob = probs[0]
    return auth_prob, fake_prob   # auth_prob, fake_prob


def predict_sklearn(text):
    proba = model.predict_proba([text])[0]
    return proba[0], proba[1]   # auth_prob, fake_prob


# ── Sample Reviews ─────────────────────────────────────────────────────────────
SAMPLE_FAKE = (
    "This product is absolutely amazing! I LOVE it so much! "
    "It is the BEST thing I have ever bought in my entire life! "
    "The quality is top notch and the material feels incredibly premium. "
    "I have already recommended this to all my friends and family members. "
    "Buy it now, you will not regret it one bit! 10 out of 10, absolutely perfect!"
)

SAMPLE_AUTHENTIC = (
    "The wireless earbuds arrived on time and well packaged. "
    "I have been using this for about three weeks now. "
    "The build quality is decent but not exceptional — the plastic parts feel a bit hollow. "
    "Battery life is about eighty percent of what they claim, which is still acceptable. "
    "The app that pairs with it is a bit clunky and could use a serious design update. "
    "Overall I would recommend it with the caveats mentioned above."
)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 Review Forensics AI")
    st.markdown("---")

    # Model badge
    if model_type == "bert":
        st.markdown('<div class="model-badge-bert">🤖 DistilBERT (Deep Learning)</div>', unsafe_allow_html=True)
    elif model_type == "sklearn":
        st.markdown('<div class="model-badge-sklearn">⚡ TF-IDF + Logistic Regression</div>', unsafe_allow_html=True)

    if meta:
        st.markdown("### 🧠 Model Stats")
        if "accuracy" in meta:
            st.markdown(f"""
            <div class="metric-box">
              <div class="metric-label">Test Accuracy</div>
              <div class="metric-value">{meta['accuracy']*100:.1f}%</div>
            </div>
            <div class="metric-box">
              <div class="metric-label">ROC-AUC Score</div>
              <div class="metric-value">{meta.get('auc', 0):.4f}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Quick Load")
    if st.button("⚠️ Load Fake Review Sample"):
        st.session_state["review_text"] = SAMPLE_FAKE
    if st.button("✅ Load Authentic Review Sample"):
        st.session_state["review_text"] = SAMPLE_AUTHENTIC

    st.markdown("---")
    st.markdown("### ℹ️ How It Works")
    if model_type == "bert":
        st.markdown("""
        1. **Tokenizer** splits text into subwords
        2. **DistilBERT** builds contextual embeddings
        3. **Classifier head** predicts Fake / Authentic
        4. **LIME** explains word-level decisions
        """)
    else:
        st.markdown("""
        1. **TF-IDF** converts text to numerical features
        2. **Logistic Regression** classifies the review
        3. **LIME** explains which words drove the decision
        """)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.75rem;color:#484f58;text-align:center;">
        NLP · Deep Learning · XAI · Fraud Detection
    </div>
    """, unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
model_tag = "DistilBERT" if model_type == "bert" else "TF-IDF"
st.markdown(f"""
<div class="header-card">
  <p class="header-title">🔍 Review Forensics AI</p>
  <p class="header-subtitle">
    An AI-powered forensic tool that detects fake, deceptive, and AI-generated
    e-commerce reviews using <b>{model_tag}</b> and Explainable AI (LIME).
  </p>
</div>
""", unsafe_allow_html=True)

if model_type is None:
    st.error("⚠️ No model found! Please run `python train_bert.py` or `python train_model.py` first.")
    st.stop()


# ── Input Area ─────────────────────────────────────────────────────────────────
st.markdown("### 📝 Paste Review Text")
review_text = st.text_area(
    label="review_input",
    label_visibility="collapsed",
    placeholder="Paste an Amazon, Flipkart, or any e-commerce product review here…",
    height=160,
    key="review_text",
)

analyze_btn = st.button("🔬 Analyze Review", use_container_width=True)


# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze_btn:
    text = review_text.strip()
    if not text:
        st.warning("⚠️ Please paste a review before analyzing.")
    else:
        with st.spinner("🔄 Running forensic analysis …"):

            # ── Prediction ────────────────────────────────────────────────────
            if model_type == "bert":
                auth_pct_raw, fake_pct_raw = predict_bert(text)
            else:
                auth_pct_raw, fake_pct_raw = predict_sklearn(text)

            fake_pct = fake_pct_raw * 100
            auth_pct = auth_pct_raw * 100
            is_fake  = fake_pct >= 50

            # ── Verdict ───────────────────────────────────────────────────────
            if is_fake:
                st.markdown(f"""
                <div class="result-fake">
                  <p class="verdict-text verdict-fake">⚠️ LIKELY FAKE / DECEPTIVE</p>
                  <p class="verdict-sub">
                    The model is <strong>{fake_pct:.1f}%</strong> confident this review
                    exhibits deceptive patterns. Exercise caution.
                  </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-authentic">
                  <p class="verdict-text verdict-authentic">✅ LIKELY AUTHENTIC</p>
                  <p class="verdict-sub">
                    The model is <strong>{auth_pct:.1f}%</strong> confident this review
                    appears genuine and trustworthy.
                  </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### 📊 Authenticity Score")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=auth_pct,
                    number={"suffix": "%", "font": {"size": 32, "color": "#58a6ff"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#8b949e",
                                 "tickfont": {"color": "#8b949e"}},
                        "bar": {"color": "#3fb950" if not is_fake else "#f85149"},
                        "bgcolor": "#161b22",
                        "bordercolor": "#30363d",
                        "steps": [
                            {"range": [0,  40], "color": "#2d1117"},
                            {"range": [40, 65], "color": "#1c1a10"},
                            {"range": [65, 100], "color": "#0d2818"},
                        ],
                        "threshold": {
                            "line": {"color": "#58a6ff", "width": 2},
                            "thickness": 0.75, "value": 50,
                        },
                    },
                    title={"text": "Authenticity Score",
                           "font": {"color": "#8b949e", "size": 13}},
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="#0d1117", font_color="#c9d1d9",
                    margin=dict(t=60, b=20, l=20, r=20), height=280,
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                st.markdown("#### 📈 Confidence Breakdown")
                fig_bar = go.Figure(go.Bar(
                    x=["Authentic", "Fake / Deceptive"],
                    y=[auth_pct, fake_pct],
                    marker_color=["#3fb950", "#f85149"],
                    text=[f"{auth_pct:.1f}%", f"{fake_pct:.1f}%"],
                    textposition="outside",
                    textfont={"color": "#c9d1d9", "size": 14},
                ))
                fig_bar.update_layout(
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font_color="#c9d1d9",
                    yaxis=dict(range=[0, 115], gridcolor="#21262d",
                               tickfont={"color": "#8b949e"}),
                    xaxis=dict(tickfont={"color": "#c9d1d9"}),
                    margin=dict(t=40, b=20, l=20, r=20),
                    height=280, showlegend=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── LIME Explainability ───────────────────────────────────────────
            st.markdown("---")
            st.markdown("#### 🧠 AI Explanation — Which Words Triggered the Alert?")
            st.markdown("""
            <div style="font-size:0.85rem;color:#8b949e;margin-bottom:1rem;">
              Words in 🔴 <b>red</b> indicate deceptive language.
              Words in 🟢 <b>green</b> indicate authentic language.
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("Running LIME explainer (~5-10 seconds) …"):
                try:
                    from lime.lime_text import LimeTextExplainer

                    if model_type == "bert":
                        def bert_predict_proba(texts):
                            results = []
                            for t in texts:
                                a, f = predict_bert(t)
                                results.append([a, f])
                            return np.array(results)
                        predict_fn = bert_predict_proba
                    else:
                        predict_fn = model.predict_proba

                    explainer = LimeTextExplainer(class_names=["Authentic", "Fake"])
                    exp = explainer.explain_instance(
                        text, predict_fn,
                        num_features=15, num_samples=300,
                    )
                    word_weights = dict(exp.as_list())

                    # Build highlighted HTML
                    words    = text.split()
                    html_out = '<div class="highlighted-text">'
                    for word in words:
                        clean  = word.strip(".,!?;:\"'()[]").lower()
                        weight = word_weights.get(clean, 0)
                        if weight > 0.01:
                            html_out += f'<span class="word-fake">{word}</span> '
                        elif weight < -0.01:
                            html_out += f'<span class="word-authentic">{word}</span> '
                        else:
                            html_out += f'<span class="word-neutral">{word}</span> '
                    html_out += "</div>"

                    st.markdown(
                        f'<div class="explain-panel">'
                        f'<div class="explain-title">🔬 Forensic Word Analysis</div>'
                        f'{html_out}</div>',
                        unsafe_allow_html=True
                    )

                    # Top keyword bar chart
                    sorted_words = sorted(
                        word_weights.items(), key=lambda x: abs(x[1]), reverse=True
                    )[:12]
                    words_list   = [w[0] for w in sorted_words]
                    weights_list = [w[1] for w in sorted_words]
                    colors       = ["#f85149" if w > 0 else "#3fb950" for w in weights_list]

                    fig_lime = go.Figure(go.Bar(
                        x=weights_list, y=words_list, orientation="h",
                        marker_color=colors,
                        text=[f"{w:+.3f}" for w in weights_list],
                        textposition="outside",
                        textfont={"color": "#c9d1d9", "size": 11},
                    ))
                    fig_lime.update_layout(
                        title=dict(text="Top Keywords by Forensic Weight",
                                   font={"color": "#8b949e", "size": 13}),
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font_color="#c9d1d9",
                        xaxis=dict(gridcolor="#21262d",
                                   tickfont={"color": "#8b949e"},
                                   title="← Authentic  |  Fake →",
                                   title_font={"color": "#8b949e"}),
                        yaxis=dict(tickfont={"color": "#c9d1d9"}),
                        margin=dict(t=40, b=20, l=20, r=80),
                        height=380, showlegend=False,
                    )
                    st.plotly_chart(fig_lime, use_container_width=True)

                except Exception as e:
                    st.warning(f"LIME explanation unavailable: {e}")

            st.markdown("---")
            st.markdown("""
            <div style="font-size:0.8rem;color:#484f58;text-align:center;padding:1rem 0;">
                🔍 Review Forensics AI &nbsp;|&nbsp; Deep Learning + Explainable AI
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#484f58;">
        <div style="font-size:3rem;">🕵️</div>
        <div style="font-size:1.1rem;margin-top:1rem;color:#8b949e;">
            Paste a review above and click <b>Analyze Review</b> to begin forensic analysis.
        </div>
        <div style="font-size:0.85rem;margin-top:0.5rem;">
            Or use the sidebar buttons to load a sample review.
        </div>
    </div>
    """, unsafe_allow_html=True)
