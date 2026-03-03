# ==========================================
# Portfolio Assignment 2 (M4)
# Green Patent Detection (PatentSBERTa)
# Active Learning + LLM → Human HITL
#
# Author: Roger Braun
# Environment: AAU Lab
# ==========================================

import os
import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


# ==========================================
# Folder Setup
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
EMB_DIR = os.path.join(BASE_DIR, "embeddings")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(EMB_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ==========================================
# Utility Printing
# ==========================================

def section(title):
    print("\n" + "=" * 70)
    print(f"🚀 {title}")
    print("=" * 70 + "\n")

def info(msg):
    print(f"📌 {msg}")

def success(msg):
    print(f"✅ {msg}")

def warning(msg):
    print(f"⚠️  {msg}")


# ==========================================
# Reproducibility
# ==========================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================
# Load Dataset
# ==========================================

section("Loading Dataset Splits")

dataset = load_from_disk(
    os.path.join(DATA_DIR, "patents_50k_green_splits")
)

train_silver = dataset["train_silver"]
eval_silver = dataset["eval_silver"]
pool_unlabeled = dataset["pool_unlabeled"]

info(f"Train size: {len(train_silver)}")
info(f"Eval size: {len(eval_silver)}")
info(f"Pool size: {len(pool_unlabeled)}")
success("Dataset loaded successfully.")


# ==========================================
# Load PatentSBERTa (Frozen Encoder)
# ==========================================

section("Loading PatentSBERTa (Frozen Encoder)")

model_name = "AI-Growth-Lab/PatentSBERTa"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

model.eval()
for param in model.parameters():
    param.requires_grad = False

success("Encoder frozen.")


# ==========================================
# Add Silver Labels
# ==========================================

section("Adding Silver Labels")

y02_cols = [col for col in train_silver.column_names if col.startswith("Y02")]

def add_green_label(example):
    example["is_green_silver"] = int(
        any(example[col] == 1 for col in y02_cols)
    )
    return example

train_silver = train_silver.map(add_green_label)
eval_silver = eval_silver.map(add_green_label)

success("Silver labels added.")


# ==========================================
# Embedding Function
# ==========================================

def get_embeddings(texts, batch_size=32):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]

        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts

        embeddings.append(mean_pooled.numpy())

    return np.vstack(embeddings)


# ==========================================
# Part A — Baseline
# ==========================================

section("Part A — Baseline Model")

train_emb_path = os.path.join(EMB_DIR, "X_train.npy")
eval_emb_path = os.path.join(EMB_DIR, "X_eval.npy")

if os.path.exists(train_emb_path):
    warning("Using cached X_train.npy")
    X_train = np.load(train_emb_path)
else:
    X_train = get_embeddings(train_silver["text"])
    np.save(train_emb_path, X_train)

if os.path.exists(eval_emb_path):
    warning("Using cached X_eval.npy")
    X_eval = np.load(eval_emb_path)
else:
    X_eval = get_embeddings(eval_silver["text"])
    np.save(eval_emb_path, X_eval)

y_train = np.array(train_silver["is_green_silver"])
y_eval = np.array(eval_silver["is_green_silver"])

clf = LogisticRegression(max_iter=1000, class_weight="balanced")
clf.fit(X_train, y_train)

y_pred = clf.predict(X_eval)

print("\n📊 Baseline Evaluation")
print(classification_report(y_eval, y_pred))


# ==========================================
# Part B — Uncertainty Sampling
# ==========================================

section("Part B — Uncertainty Sampling")

pool_emb_path = os.path.join(EMB_DIR, "X_pool.npy")

if os.path.exists(pool_emb_path):
    warning("Using cached X_pool.npy")
    X_pool = np.load(pool_emb_path)
else:
    X_pool = get_embeddings(pool_unlabeled["text"])
    np.save(pool_emb_path, X_pool)

p_green = clf.predict_proba(X_pool)[:, 1]
u = 1 - 2 * np.abs(p_green - 0.5)
top_indices = np.argsort(-u)[:100]

df_hitl = pd.DataFrame({
    "doc_id": pool_unlabeled.select(top_indices)["id"],
    "text": pool_unlabeled.select(top_indices)["text"],
    "p_green": p_green[top_indices],
    "u": u[top_indices],
    "llm_green_suggested": "",
    "llm_confidence": "",
    "llm_rationale": "",
    "is_green_human": ""
})

hitl_ready_path = os.path.join(OUTPUT_DIR, "hitl_review_ready.tsv")
df_hitl.to_csv(hitl_ready_path, sep="\t", index=False)

success("HITL file exported.")
print("👉 Open outputs/hitl_review_ready.tsv")
print("👉 Label is_green_human")
print("👉 Save as outputs/hitl_reviewed.csv")


# ==========================================
# STOP IF HITL NOT DONE
# ==========================================

hitl_reviewed_path = os.path.join(OUTPUT_DIR, "hitl_reviewed.csv")

if not os.path.exists(hitl_reviewed_path):
    warning("hitl_reviewed.csv not found.")
    exit()

hitl_df = pd.read_csv(hitl_reviewed_path, sep="\t")


# ==========================================
# X_hitl Embeddings
# ==========================================

hitl_emb_path = os.path.join(EMB_DIR, "X_hitl.npy")

if os.path.exists(hitl_emb_path):
    X_hitl = np.load(hitl_emb_path)
else:
    X_hitl = get_embeddings(hitl_df["text"].tolist())
    np.save(hitl_emb_path, X_hitl)


# ==========================================
# Merge Gold Labels
# ==========================================

section("Merging Gold Labels")

gold_dict = dict(zip(hitl_df["doc_id"], hitl_df["is_green_human"]))

def override_label(example):
    if example["id"] in gold_dict:
        example["is_green_gold"] = gold_dict[example["id"]]
    else:
        example["is_green_gold"] = example["is_green_silver"]
    return example

train_gold = train_silver.map(override_label)

train_final = train_gold.remove_columns(
    [col for col in train_gold.column_names if col not in ["text", "is_green_gold"]]
)

eval_final = eval_silver.remove_columns(
    [col for col in eval_silver.column_names if col not in ["text", "is_green_silver"]]
)

train_final = train_final.rename_column("is_green_gold", "label")
eval_final = eval_final.rename_column("is_green_silver", "label")


# ==========================================
# Fine-Tuning
# ==========================================

section("Fine-Tuning PatentSBERTa")

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

train_final = train_final.map(tokenize, batched=True)
eval_final = eval_final.map(tokenize, batched=True)

train_final.set_format("torch")
eval_final.set_format("torch")

training_args = TrainingArguments(
    output_dir=os.path.join(MODEL_DIR, "patentsberta_final"),
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=50,
    ddp_find_unused_parameters=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_final,
    eval_dataset=eval_final,
    compute_metrics=compute_metrics
)

trainer.train()

print("\n📊 Evaluation on eval_silver")
print(trainer.evaluate())

success("Pipeline finished successfully 🎉")