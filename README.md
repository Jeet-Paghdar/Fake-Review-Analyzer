# 🔍 Fake Review Forensic Analyzer

> **An AI-powered forensic tool that detects fake, deceptive, and AI-generated e-commerce reviews using Natural Language Processing (NLP) and Explainable AI (LIME).**

---

## 🚨 The Real-World Problem

The global e-commerce market loses billions of dollars annually to fake reviews. Platforms like Amazon, Flipkart, and Yelp are flooded with AI-generated or incentivized 5-star reviews that:
- Mislead genuine buyers into purchasing poor-quality products
- Suppress honest sellers with authentic reviews
- Erode consumer trust across the entire ecosystem

Traditional keyword-based filters are easily bypassed. **This project uses machine learning to detect the underlying linguistic fingerprints of deceptive writing.**

---

## 🧠 How It Works

```
Review Text
    │
    ▼
TF-IDF Vectorizer (Bigrams, 15,000 features)
    │
    ▼
Logistic Regression Classifier
    │
    ▼
Prediction: Authentic / Fake + Confidence Score
    │
    ▼
LIME Explainer → Highlights suspicious words
```

### Key ML Concepts Demonstrated
| Concept | Implementation |
|---|---|
| **Text Vectorization** | TF-IDF with unigram + bigram features |
| **Classification** | Logistic Regression with balanced class weights |
| **Model Evaluation** | Accuracy, ROC-AUC, 5-fold Cross-Validation |
| **Explainable AI (XAI)** | LIME (Local Interpretable Model-agnostic Explanations) |
| **Deployment** | Interactive Streamlit web dashboard |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Scikit-learn** — TF-IDF, Logistic Regression, Pipeline
- **LIME** — Explainable AI for text
- **Streamlit** — Interactive web dashboard
- **Plotly** — Gauge and bar chart visualizations

---

## 🚀 Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/fake_review_analyzer.git
cd fake_review_analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate dataset & train the model
```bash
python generate_data.py   # Creates reviews_dataset.csv (2400 samples)
python train_model.py     # Trains and saves review_model.pkl
```

### 5. Launch the web app
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Test Accuracy | ~97% |
| ROC-AUC | ~0.99 |
| Cross-Validation | 5-fold |

---

## 🖥️ Features

- **📝 Review Input** — Paste any e-commerce review text
- **🔬 Verdict** — Instant Fake/Authentic prediction with confidence score
- **📊 Gauge Chart** — Visual Authenticity Score (0–100%)
- **📈 Confidence Breakdown** — Side-by-side bar chart
- **🧠 LIME Explainability** — Word-level forensic highlighting (red = suspicious, green = authentic)
- **📉 Keyword Weight Chart** — Top words that influenced the decision
- **🎛️ Sample Reviews** — Pre-loaded Fake/Authentic examples in the sidebar

---

## 📁 Project Structure

```
fake_review_analyzer/
│
├── generate_data.py     # Synthetic dataset generation
├── train_model.py       # ML training pipeline
├── app.py               # Streamlit dashboard
├── requirements.txt     # Dependencies
├── reviews_dataset.csv  # Generated after running generate_data.py
├── review_model.pkl     # Saved model (generated after training)
├── model_meta.pkl       # Model metadata
└── README.md
```

---

## 💡 Future Improvements

- [ ] Fine-tune a DistilBERT transformer for higher accuracy
- [ ] Add support for analyzing reviews in regional languages
- [ ] Integrate with live Amazon/Flipkart product URLs via web scraping
- [ ] Deploy to Streamlit Cloud for public access


