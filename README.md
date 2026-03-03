Portfolio Assignment 2 (M4)
Green Patent Detection (PatentSBERTa)

Active Learning + LLM → Human HITL

Course: Applied Deep Learning
Module: M4
Deadline: Monday 16 February 2026, 12:00
Instructor: Hamid B.
Student: Roger Braun

📌 Overview

This assignment implements a complete green patent classification pipeline using PatentSBERTa.

The workflow follows four sequential stages:

Part A – Baseline Model (Frozen Embeddings)

Part B – Uncertainty Sampling

Part C – LLM → Human-in-the-Loop (HITL)

Part D – Final Fine-Tuning

The objective is to improve label quality and classification performance through active learning and gold supervision.

🧱 Part A – Baseline Model (Frozen Embeddings)
Description

A fast baseline classifier is trained using frozen PatentSBERTa embeddings and Logistic Regression.

The encoder is frozen:

Embeddings
=
𝑓
PatentSBERTa
(
𝑥
)
Embeddings=f
PatentSBERTa
	​

(x)

Only the classifier is trained.

Dataset

Dataset: AI-Growth-Lab/patents_claims_1.5m_traim_test (50k balanced sample)

File: patents_50k_green.parquet

Splits:

train_silver (30,000)

eval_silver (10,000)

pool_unlabeled (10,000)

The dataset is class-balanced (~50% green / ~50% non-green).

Silver Label

Green label derived from CPC Y02* codes:

is_green_silver = 1 if any Y02 code present
Evaluation

Metrics reported:

Precision

Recall

F1-score

Accuracy

Baseline performance ≈ 0.78 F1.

🎯 Part B – Identify High-Risk Examples (Uncertainty Sampling)
Definition

Uncertainty score:

𝑢
=
1
−
2
∣
𝑝
−
0.5
∣
u=1−2∣p−0.5∣

Where:

p = predicted probability of green

High-risk examples are those with highest u.

Procedure

Compute probabilities for all pool_unlabeled samples

Compute uncertainty

Select top 100

Export to:

outputs/hitl_review_ready.tsv

Exported fields:

doc_id

text

p_green

u

llm_green_suggested

llm_confidence

llm_rationale

is_green_human

No CPC or metadata used for selection.

🤝 Part C – LLM → Human HITL
Workflow

For each of the 100 uncertain samples:

LLM evaluates claim text only

Outputs:

llm_green_suggested (0/1)

llm_confidence (low/medium/high)

llm_rationale (1–3 sentences)

Human assigns final label:

is_green_human (0/1)

Gold labels override silver labels.

Gold Merge Rule
is_green_gold = human label if available
otherwise = silver label
🚀 Part D – Final Model (Fine-Tune PatentSBERTa Once)

PatentSBERTa is fine-tuned end-to-end using:

train_silver + gold_100

Binary classification

Hyperparameters

max_seq_length = 256

learning_rate = 2e-5

batch_size = 16

epochs = 1

weight_decay = 0.01

Evaluation performed on:

eval_silver

gold_100

▶️ How to Run
Local
python assignment02_m4.py
SLURM
sbatch assignment02_m4.sh
📦 Deliverables

Fine-tuned model uploaded to Hugging Face Hub

Gold-enhanced dataset uploaded to Hugging Face Hub

Video demonstration

Email sent to instructor with links