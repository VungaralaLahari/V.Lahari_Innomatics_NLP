# NLP Task 5 — Fine-Tuning DistilBERT for Token Classification

## What is this project?

This project fine-tunes a pre-trained **DistilBERT** model to label every word in a sentence with a tag.

For example:
```
Input  : John works at Google in California
Output : B-PER  O      O  B-ORG  O  B-LOC
```

Every word gets a tag — Person, Organization, Location, or Other (O).
This is called **Token Classification**.

---

## Dataset

**WikiANN (English)**

- 20,000 training sentences
- 10,000 validation sentences
- 10,000 test sentences
- Labels: `O`, `B-PER`, `I-PER`, `B-ORG`, `I-ORG`, `B-LOC`, `I-LOC`

> CoNLL-2003 was originally planned but had compatibility issues with the current
> HuggingFace datasets library. WikiANN uses the same IOB token classification
> format and fully covers the assignment requirements.

---

## What is IOB Format?

- **B-** → Beginning of an entity ("New" in "New York")
- **I-** → Inside an entity ("York" in "New York")
- **O** → Not an entity (regular word)

---

## Tools Used

- Python
- HuggingFace Transformers
- HuggingFace Datasets
- PyTorch
- seqeval (for evaluation)
- Google Colab (GPU)

---

## Pipeline

```
Raw Data → Tokenization → Label Alignment → Model Training → Evaluation → Inference
```

---

## The Hardest Part — Label Alignment

DistilBERT breaks words into subwords sometimes.

Example:
```
"Saunders" → ["Sau", "##nders"]   ← 2 tokens, 1 label
```

Only the **first subword gets the real label**.
The rest get **-100** so the model ignores them during training.

---

## Results

| Metric    | Score  |
|-----------|--------|
| Precision | ~0.84  |
| Recall    | ~0.85  |
| F1 Score  | ~0.84  |
| Accuracy  | ~0.97  |

> Actual scores may vary slightly based on runtime and GPU.

---

## How to Run

1. Open `Task5_NLP_TokenClassification.ipynb` in Google Colab
2. Enable GPU — Runtime → Change runtime type → T4 GPU
3. Run all cells top to bottom
4. Training takes around 15–20 minutes

---

## Note

This was my first time working with transformer models for token classification.
I understood the preprocessing and model building parts well.
The training and evaluation parts were done with guidance.
