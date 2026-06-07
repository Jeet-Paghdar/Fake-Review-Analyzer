"""
train_bert.py
Fine-tunes DistilBERT for fake review detection.
Recommended: Run on Google Colab (free GPU) for ~15 min training.
Local CPU training will take 1-2 hours.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score
)
from generate_data import generate_dataset

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LEARNING_RATE = 2e-5
SAVE_DIR     = "bert_model"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Dataset Class ──────────────────────────────────────────────────────────────
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Training Function ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["label"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ── Evaluation Function ────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0, [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            probs  = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds  = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    auc      = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, auc, all_labels, all_preds


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Fake Review Forensic Analyzer — DistilBERT Training")
    print("=" * 60)
    print(f"\n🖥️  Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"   GPU    : {torch.cuda.get_device_name(0)}")
    else:
        print("   ⚠️  No GPU found. Training on CPU will be slow (~1-2 hrs).")
        print("   💡 Tip : Upload this file to Google Colab for free GPU.")

    # ── 1. Load Data ─────────────────────────────────────────────────────────
    if os.path.exists("reviews_dataset.csv"):
        print("\n📂 Loading existing dataset …")
        df = pd.read_csv("reviews_dataset.csv")
    else:
        print("\n⚙️  Generating synthetic dataset …")
        df = generate_dataset()
        df.to_csv("reviews_dataset.csv", index=False)

    print(f"   Total  : {len(df)} samples")
    print(f"   Fake   : {df['label'].sum()}")
    print(f"   Authentic : {(df['label']==0).sum()}")

    texts  = df["review"].tolist()
    labels = df["label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    print(f"\n   Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

    # ── 2. Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n📥 Loading tokenizer: {MODEL_NAME} …")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    train_dataset = ReviewDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_dataset   = ReviewDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    test_dataset  = ReviewDataset(X_test,  y_test,  tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    # ── 3. Model ─────────────────────────────────────────────────────────────
    print(f"📥 Loading model: {MODEL_NAME} …")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(DEVICE)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params    : {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    # ── 4. Optimizer & Scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # ── 5. Training Loop ─────────────────────────────────────────────────────
    print(f"\n🚀 Starting training for {EPOCHS} epochs …\n")
    best_val_acc = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"── Epoch {epoch}/{EPOCHS} " + "─" * 40)
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader)

        print(f"   Train → Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"   Val   → Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  AUC: {val_auc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(SAVE_DIR, exist_ok=True)
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"   ✅ Best model saved → {SAVE_DIR}/")

    # ── 6. Final Test Evaluation ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Final Test Set Evaluation")
    print("=" * 60)

    # Reload best model
    best_model = DistilBertForSequenceClassification.from_pretrained(SAVE_DIR)
    best_model.to(DEVICE)

    _, test_acc, test_auc, y_true, y_pred = evaluate(best_model, test_loader)

    print(f"\n📊 Test Accuracy : {test_acc:.4f}")
    print(f"   ROC-AUC      : {test_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Authentic", "Fake"]))

    # Save metadata
    import joblib
    meta = {
        "accuracy":   test_acc,
        "auc":        test_auc,
        "model_type": "DistilBERT",
        "classes":    ["Authentic", "Fake"],
    }
    joblib.dump(meta, "model_meta.pkl")
    print("\n✅ Metadata saved → model_meta.pkl")
    print("🎉 Training complete! Run → streamlit run app.py")


if __name__ == "__main__":
    main()
# Finalized DistilBERT configuration
