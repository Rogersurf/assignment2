# 🚀 Portfolio Assignment 2 (M4)  
## 🌱 Green Patent Detection with PatentSBERTa  
### Active Learning + LLM → Human-in-the-Loop (HITL)

**Institution:** Aalborg University
**Module:** Applied Deep Learning and Artificial Intelligence
**Instructor:** Hamid B.  
**Student:** Roger Braun  
**published:** Monday, 16 February 2026  

---

# 📌 Overview

This project implements a complete green patent classification pipeline using PatentSBERTa and an Active Learning workflow.

The pipeline consists of four stages:

- 🧱 Part A — Baseline Model (Frozen Embeddings)  
- 🎯 Part B — Uncertainty Sampling  
- 🤝 Part C — LLM → Human-in-the-Loop (HITL)  
- 🚀 Part D — Final Fine-Tuning  

The objective is to improve label quality and classification performance using uncertainty-driven selection and gold supervision.

---

# 🧱 Part A — Baseline Model (Frozen Embeddings)

## 📖 Description

A fast baseline classifier is trained using frozen PatentSBERTa embeddings and a Logistic Regression classifier.

The encoder is frozen:

Embeddings = PatentSBERTa(x)

Only the classifier parameters are trained.

---

## 📊 Dataset

- Source: AI-Growth-Lab/patents_claims_1.5m_traim_test  
- Local file: patents_50k_green.parquet  
- Balanced sample: 50,000 claims  
  - 25,000 green (CPC Y02*)  
  - 25,000 non-green  

### Splits

- train_silver — 30,000  
- eval_silver — 10,000  
- pool_unlabeled — 10,000  

The dataset is approximately class-balanced (~50/50).

---

## 🏷 Silver Label Definition

Green label is derived from CPC Y02* codes:

```python
is_green_silver = 1 if any(Y02) else 0
```

---

## 📈 Baseline Evaluation Metrics

- Precision  
- Recall  
- F1-score  
- Accuracy  

Baseline performance:

**F1 ≈ 0.78**

---

# 🎯 Part B — Identify High-Risk Examples (Uncertainty Sampling)

## 📖 Definition

Let:

p = predicted probability that a claim is green

Uncertainty score:

u = 1 - 2 * abs(p - 0.5)

Where:

- u = 1 → most uncertain (p = 0.5)  
- u ≈ 0 → confident prediction  

---

## 🔍 Procedure

1. Compute p_green for all samples in pool_unlabeled  
2. Compute uncertainty score u  
3. Select the top 100 highest-u examples  
4. Export to:

```
outputs/hitl_review_ready.tsv
```

---

## 📁 Exported Fields

- doc_id  
- text  
- p_green  
- u  
- llm_green_suggested  
- llm_confidence  
- llm_rationale  
- is_green_human  

Selection is based only on model uncertainty.

---

# 🤝 Part C — LLM → Human-in-the-Loop (HITL)

## 📖 Workflow

For each of the 100 uncertain samples:

### 🔹 LLM Evaluation

Given only the claim text:

- llm_green_suggested (0/1)  
- llm_confidence (low/medium/high)  
- llm_rationale (1–3 sentences citing the claim)

### 🔹 Human Review (Final Label)

The human reviewer assigns:

- is_green_human (0/1)

The human decision overrides the LLM suggestion.

---

## 🥇 Gold Label Merge Rule

is_green_gold = human_label if reviewed else silver_label

This produces a gold-enhanced training dataset.

---

# 🚀 Part D — Final Model (Fine-Tune PatentSBERTa Once)

PatentSBERTa is fine-tuned end-to-end using:

- train_silver + gold_100  
- Binary classification  

---

## ⚙️ Hyperparameters

- max_seq_length = 256  
- learning_rate = 2e-5  
- batch_size = 16  
- epochs = 1  
- weight_decay = 0.01  

---

## 📊 Evaluation

Performance is reported on:

- eval_silver  
- gold_100  

Metrics:

- Accuracy  
- Precision  
- Recall  
- F1-score  

---

# ▶️ How to Run

## 💻 Local

```bash
python assignment02_m4.py
```

## 🖥 SLURM

```bash
sbatch assignment02_m4.sh
```

---

# 📦 Deliverables

- Fine-tuned model uploaded to Hugging Face Hub  
- Gold-enhanced dataset uploaded to Hugging Face Hub  
- Video demonstration  
- Email sent to instructor with links  

---

# 🧠 Key Learning Contributions

This assignment demonstrates:

- Transfer learning with frozen embeddings  
- Uncertainty-based Active Learning  
- LLM-assisted labeling  
- Human supervision integration  
- End-to-end transformer fine-tuning  