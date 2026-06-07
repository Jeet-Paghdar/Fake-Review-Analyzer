"""
create_notebook.py
Run this script once to generate the Google Colab notebook.
Usage: python create_notebook.py
It will create: train_bert.ipynb
Dataset: Kaggle — Amazon Fake Reviews Dataset (~21,000 reviews)
"""

import json

notebook = {
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [

    # ── Cell 1: Title ─────────────────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# 🔍 Fake Review Forensic Analyzer — DistilBERT Training\n",
        "\n",
        "> Fine-tuning DistilBERT on the **Kaggle Amazon Fake Reviews Dataset** (~21,000 real reviews).\n",
        "\n",
        "### ⚡ Before Running:\n",
        "1. Go to **Runtime → Change runtime type → T4 GPU**\n",
        "2. Make sure you have copied your **Kaggle API Token** to your clipboard.\n",
        "\n",
        "### Steps:\n",
        "1. Setup Kaggle API (Paste your token)\n",
        "2. Download & explore dataset\n",
        "3. Preprocess data\n",
        "4. Fine-tune DistilBERT (~15 min on T4 GPU)\n",
        "5. Evaluate model\n",
        "6. Download trained model to your PC\n"
      ]
    },

    # ── Cell 2: GPU Check ─────────────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## ⚡ Step 0 — Check GPU\n",
                 "> Make sure GPU is enabled: **Runtime → Change runtime type → T4 GPU**"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import torch\n",
        "print('CUDA available:', torch.cuda.is_available())\n",
        "if torch.cuda.is_available():\n",
        "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n",
        "    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n",
        "else:\n",
        "    print('⚠️  No GPU detected!')\n",
        "    print('Go to Runtime → Change runtime type → T4 GPU and restart.')\n"
      ],
      "outputs": [],
      "execution_count": None
    },

    # ── Cell 3: Kaggle Setup ──────────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📦 Step 1 — Setup Kaggle API\n",
        "\n",
        "Run the cell below and **paste the Kaggle API Token** you copied from the Kaggle settings page.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import os\n",
        "from getpass import getpass\n",
        "\n",
        "print('🔑 Please paste your Kaggle API Token below and press Enter:')\n",
        "token = getpass('Token: ')\n",
        "os.environ['KAGGLE_API_TOKEN'] = token.strip()\n",
        "print('✅ Kaggle API configured successfully!')\n"
      ],
      "outputs": [],
      "execution_count": None
    },

    # ── Cell 4: Install ───────────────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## 📦 Step 2 — Install Dependencies"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Install required packages\n",
        "!pip install -q --upgrade transformers\n",
        "!pip install -q kaggle scikit-learn pandas numpy lime plotly joblib\n",
        "print('✅ All packages installed!')\n",
        "print()\n",
        "print('=' * 50)\n",
        "print('NEXT ACTION REQUIRED:')\n",
        "print('1. Click Runtime in the top menu')\n",
        "print('2. Click Restart session')\n",
        "print('3. Click Runtime → Run all')\n",
        "print('   (This cell will be skipped on the next run)')\n",
        "print('=' * 50)\n"
      ],
      "outputs": [],
      "execution_count": None
    },

    # ── Cell 5: Download Dataset ──────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📊 Step 3 — Download Kaggle Dataset\n",
        "> Dataset: **Amazon Fake Reviews** by mexwell (~21,000 reviews)\n",
        "> Columns: `text_` (review text), `label` (CG=Fake, OR=Original/Real)\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Download dataset from Kaggle\n",
        "!kaggle datasets download -d mexwell/fake-reviews-dataset --unzip\n",
        "\n",
        "import os, glob\n",
        "csv_files = glob.glob('*.csv')\n",
        "print('Downloaded files:', csv_files)\n"
      ],
      "outputs": [],
      "execution_count": None
    },

    # ── Cell 6: Explore & Preprocess ─────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## 🔍 Step 4 — Explore & Preprocess Dataset"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "# Load dataset\n",
        "df = pd.read_csv('fake reviews dataset.csv')\n",
        "print('Shape:', df.shape)\n",
        "print('\\nColumns:', df.columns.tolist())\n",
        "print('\\nFirst few rows:')\n",
        "print(df.head())\n",
        "print('\\nLabel distribution:')\n",
        "print(df['label'].value_counts())\n"
      ],
      "outputs": [],
      "execution_count": None
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "# Preprocess\n",
        "# label: CG = Computer Generated (FAKE=1), OR = Original Real (AUTHENTIC=0)\n",
        "df = df[['text_', 'label']].dropna()\n",
        "df.columns = ['review', 'label_raw']\n",
        "df['label'] = df['label_raw'].map({'CG': 1, 'OR': 0})\n",
        "df = df.dropna(subset=['label'])\n",
        "df['label'] = df['label'].astype(int)\n",
        "\n",
        "# Remove very short reviews (< 20 chars)\n",
        "df = df[df['review'].str.len() > 20].reset_index(drop=True)\n",
        "\n",
        "# Balance dataset (equal fake and authentic)\n",
        "min_count = min(df['label'].value_counts())\n",
        "df = pd.concat([\n",
        "    df[df['label'] == 0].sample(min_count, random_state=42),\n",
        "    df[df['label'] == 1].sample(min_count, random_state=42),\n",
        "]).sample(frac=1, random_state=42).reset_index(drop=True)\n",
        "\n",
        "print(f'✅ Final dataset: {len(df)} samples')\n",
        "print(f'   Fake (CG)      : {(df[\"label\"]==1).sum()}')\n",
        "print(f'   Authentic (OR) : {(df[\"label\"]==0).sum()}')\n",
        "print(f'\\nSample fake review:')\n",
        "print(df[df['label']==1]['review'].iloc[0][:300])\n",
        "print(f'\\nSample authentic review:')\n",
        "print(df[df['label']==0]['review'].iloc[0][:300])\n",
        "\n",
        "df.to_csv('reviews_dataset.csv', index=False)\n",
        "print('\\n✅ Saved as reviews_dataset.csv')\n"
      ],
      "outputs": [],
      "execution_count": None
    },

    # ── Cell 7: Train ─────────────────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🚀 Step 5 — Fine-Tune DistilBERT\n",
        "> ⏳ Expected time: **~15-20 minutes** on T4 GPU\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import torch\n",
        "from torch.utils.data import Dataset, DataLoader\n",
        "from transformers import (\n",
        "    DistilBertTokenizerFast,\n",
        "    DistilBertForSequenceClassification,\n",
        "    get_linear_schedule_with_warmup,\n",
        ")\n",
        "from torch.optim import AdamW\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.metrics import accuracy_score, classification_report, roc_auc_score\n",
        "import joblib, os\n",
        "\n",
        "# ── Config ───────────────────────────────────────────────────────────────────\n",
        "MODEL_NAME    = 'distilbert-base-uncased'\n",
        "MAX_LEN       = 256\n",
        "BATCH_SIZE    = 16\n",
        "EPOCHS        = 3\n",
        "LEARNING_RATE = 2e-5\n",
        "SAVE_DIR      = 'bert_model'\n",
        "DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
        "print(f'Using device: {DEVICE}')\n",
        "\n",
        "# ── Dataset class ─────────────────────────────────────────────────────────────\n",
        "class ReviewDataset(Dataset):\n",
        "    def __init__(self, texts, labels, tokenizer, max_len):\n",
        "        self.texts = texts\n",
        "        self.labels = labels\n",
        "        self.tokenizer = tokenizer\n",
        "        self.max_len = max_len\n",
        "    def __len__(self):\n",
        "        return len(self.texts)\n",
        "    def __getitem__(self, idx):\n",
        "        enc = self.tokenizer(\n",
        "            self.texts[idx], max_length=self.max_len,\n",
        "            padding='max_length', truncation=True, return_tensors='pt'\n",
        "        )\n",
        "        return {\n",
        "            'input_ids':      enc['input_ids'].squeeze(0),\n",
        "            'attention_mask': enc['attention_mask'].squeeze(0),\n",
        "            'label':          torch.tensor(self.labels[idx], dtype=torch.long),\n",
        "        }\n",
        "\n",
        "# ── Load data ─────────────────────────────────────────────────────────────────\n",
        "df     = pd.read_csv('reviews_dataset.csv')\n",
        "texts  = df['review'].tolist()\n",
        "labels = df['label'].tolist()\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    texts, labels, test_size=0.2, random_state=42, stratify=labels)\n",
        "X_train, X_val, y_train, y_val   = train_test_split(\n",
        "    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)\n",
        "\n",
        "print(f'Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}')\n",
        "\n",
        "# ── Tokenizer & Loaders ───────────────────────────────────────────────────────\n",
        "print('\\n📥 Loading DistilBERT tokenizer...')\n",
        "tokenizer    = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)\n",
        "train_loader = DataLoader(ReviewDataset(X_train, y_train, tokenizer, MAX_LEN),\n",
        "                          batch_size=BATCH_SIZE, shuffle=True)\n",
        "val_loader   = DataLoader(ReviewDataset(X_val,   y_val,   tokenizer, MAX_LEN),\n",
        "                          batch_size=BATCH_SIZE)\n",
        "test_loader  = DataLoader(ReviewDataset(X_test,  y_test,  tokenizer, MAX_LEN),\n",
        "                          batch_size=BATCH_SIZE)\n",
        "\n",
        "# ── Model, optimizer, scheduler ───────────────────────────────────────────────\n",
        "print('📥 Loading DistilBERT model...')\n",
        "model     = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)\n",
        "model.to(DEVICE)\n",
        "optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)\n",
        "total_steps = len(train_loader) * EPOCHS\n",
        "scheduler = get_linear_schedule_with_warmup(\n",
        "    optimizer,\n",
        "    num_warmup_steps=int(0.1 * total_steps),\n",
        "    num_training_steps=total_steps,\n",
        ")\n",
        "\n",
        "# ── Helper: one epoch ─────────────────────────────────────────────────────────\n",
        "def run_epoch(model, loader, optimizer=None, scheduler=None):\n",
        "    is_train = optimizer is not None\n",
        "    model.train() if is_train else model.eval()\n",
        "    total_loss, preds_all, labels_all, probs_all = 0, [], [], []\n",
        "    with torch.set_grad_enabled(is_train):\n",
        "        for batch in loader:\n",
        "            ids  = batch['input_ids'].to(DEVICE)\n",
        "            mask = batch['attention_mask'].to(DEVICE)\n",
        "            lbls = batch['label'].to(DEVICE)\n",
        "            out  = model(input_ids=ids, attention_mask=mask, labels=lbls)\n",
        "            if is_train:\n",
        "                out.loss.backward()\n",
        "                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n",
        "                optimizer.step(); scheduler.step(); optimizer.zero_grad()\n",
        "            total_loss += out.loss.item()\n",
        "            preds_all.extend(torch.argmax(out.logits, 1).cpu().detach().numpy())\n",
        "            labels_all.extend(lbls.cpu().numpy())\n",
        "            probs_all.extend(torch.softmax(out.logits, 1)[:, 1].cpu().detach().numpy())\n",
        "    acc = accuracy_score(labels_all, preds_all)\n",
        "    auc = roc_auc_score(labels_all, probs_all)\n",
        "    return total_loss / len(loader), acc, auc, labels_all, preds_all\n",
        "\n",
        "# ── Training Loop ─────────────────────────────────────────────────────────────\n",
        "best_val_acc = 0\n",
        "print(f'\\n🚀 Training for {EPOCHS} epochs on {DEVICE}...')\n",
        "print('-' * 70)\n",
        "\n",
        "for epoch in range(1, EPOCHS + 1):\n",
        "    tr_loss, tr_acc, tr_auc, _, _ = run_epoch(model, train_loader, optimizer, scheduler)\n",
        "    vl_loss, vl_acc, vl_auc, _, _ = run_epoch(model, val_loader)\n",
        "    print(f'Epoch {epoch}/{EPOCHS} | '\n",
        "          f'Train → Loss:{tr_loss:.4f} Acc:{tr_acc:.4f} | '\n",
        "          f'Val → Loss:{vl_loss:.4f} Acc:{vl_acc:.4f} AUC:{vl_auc:.4f}')\n",
        "    if vl_acc > best_val_acc:\n",
        "        best_val_acc = vl_acc\n",
        "        os.makedirs(SAVE_DIR, exist_ok=True)\n",
        "        model.save_pretrained(SAVE_DIR)\n",
        "        tokenizer.save_pretrained(SAVE_DIR)\n",
        "        print(f'  ✅ Best model checkpoint saved → {SAVE_DIR}/')\n",
        "\n",
        "print('\\n🎉 Training complete!')\n"
      ],
      "outputs": [],
      "execution_count": None
    },

    # ── Cell 8: Evaluate ──────────────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["## 📊 Step 6 — Evaluate on Test Set"]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "from transformers import DistilBertForSequenceClassification\n",
        "\n",
        "# Load best saved model\n",
        "best_model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)\n",
        "best_model.to(DEVICE)\n",
        "\n",
        "_, test_acc, test_auc, y_true, y_pred = run_epoch(best_model, test_loader)\n",
        "\n",
        "print('=' * 50)\n",
        "print(f'  Final Test Results')\n",
        "print('=' * 50)\n",
        "print(f'  Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)')\n",
        "print(f'  ROC-AUC  : {test_auc:.4f}')\n",
        "print()\n",
        "print(classification_report(y_true, y_pred, target_names=['Authentic', 'Fake']))\n",
        "\n",
        "# Save metadata\n",
        "import joblib\n",
        "meta = {\n",
        "    'accuracy':   test_acc,\n",
        "    'auc':        test_auc,\n",
        "    'model_type': 'DistilBERT',\n",
        "    'dataset':    'Kaggle Amazon Fake Reviews (~21,000 samples)',\n",
        "    'classes':    ['Authentic', 'Fake'],\n",
        "}\n",
        "joblib.dump(meta, 'model_meta.pkl')\n",
        "print('✅ model_meta.pkl saved!')\n"
      ],
      "outputs": [],
      "execution_count": None
    },

    # ── Cell 9: Download ──────────────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 📥 Step 7 — Download Trained Model to Your PC\n",
        "> This zips the entire `bert_model/` folder and downloads it automatically.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {},
      "source": [
        "import shutil\n",
        "from google.colab import files\n",
        "\n",
        "# Copy metadata into the model folder\n",
        "shutil.copy('model_meta.pkl', f'{SAVE_DIR}/model_meta.pkl')\n",
        "\n",
        "# Zip the folder\n",
        "shutil.make_archive('bert_model_download', 'zip', '.', SAVE_DIR)\n",
        "print('✅ bert_model_download.zip created!')\n",
        "print(f'   Size: {os.path.getsize(\"bert_model_download.zip\") / 1e6:.1f} MB')\n",
        "\n",
        "# Auto-download to your PC\n",
        "files.download('bert_model_download.zip')\n",
        "print('📥 Download started! Check your browser downloads.')\n"
      ],
      "outputs": [],
      "execution_count": None
    },

    # ── Cell 10: Instructions ─────────────────────────────────────────────────
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## 🎉 All Done! Next Steps\n",
        "\n",
        "After `bert_model_download.zip` downloads to your PC:\n",
        "\n",
        "### 1. Unzip it\n",
        "Extract the zip file — you will get a `bert_model/` folder.\n",
        "\n",
        "### 2. Place it in your project\n",
        "```\n",
        "fake_review_analyzer/\n",
        "├── bert_model/          ← paste the unzipped folder here\n",
        "│   ├── config.json\n",
        "│   ├── pytorch_model.bin\n",
        "│   ├── tokenizer files...\n",
        "│   └── model_meta.pkl\n",
        "├── app.py\n",
        "├── generate_data.py\n",
        "└── ...\n",
        "```\n",
        "\n",
        "### 3. Copy model_meta.pkl to root\n",
        "Also copy `bert_model/model_meta.pkl` one level up to `fake_review_analyzer/`.\n",
        "\n",
        "### 4. Launch the app\n",
        "```bash\n",
        "python -m streamlit run app.py\n",
        "```\n",
        "\n",
        "The sidebar will show **🤖 DistilBERT (Deep Learning)** — your model is live! 🚀\n"
      ]
    }

  ]
}

# Write the notebook file
with open("train_bert.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("✅ train_bert.ipynb updated successfully to use Kaggle API Token string!")
print()
print("=" * 55)
print("  NEXT STEPS")
print("=" * 55)
print()
print("1. Copy the long token string from Kaggle (it should look like 'eyJhb...')")
print("2. Upload the newly generated train_bert.ipynb to Google Colab")
print("3. Run the notebook. When asked, PASTE the token string into the text box and press Enter.")
# Finalized notebook generator structure
