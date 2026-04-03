# Fine-Tuning BERT on News Category Dataset

A text classification project built as part of an NLP assignment.  
The goal was to fine-tune a pre-trained BERT model on the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle and compare different training strategies.

---

## Dataset

**News Category Dataset** — HuffPost news articles (~200K records)  
Each article has a `headline`, `short_description`, and `category`.  
Top 10 categories were used for this project.

---

## Project Pipeline

```
Raw Data → Preprocessing → Tokenization → Model Building → Fine-Tuning → Evaluation → Experiments
```

---

## Tech Stack

- Python
- HuggingFace Transformers
- PyTorch
- Scikit-learn
- Pandas, Seaborn, Matplotlib

---

## What Was Done

### 1. Preprocessing
- Combined `headline` + `short_description` into a single `text` column
- Removed URLs, special characters, extra whitespace
- Dropped missing values and duplicates
- Kept top 10 categories and encoded labels using `LabelEncoder`

### 2. Data Splitting
| Split      | Size |
|------------|------|
| Train      | 70%  |
| Validation | 15%  |
| Test       | 15%  |

### 3. Tokenization
- Model: `bert-base-uncased`
- Max length: 128 tokens
- Padding & truncation enabled

### 4. Model
- `AutoModelForSequenceClassification` from HuggingFace
- Optimizer: `AdamW` with learning rate `2e-5`
- Epochs: 3

---

## Experiments & Results

Three experiments were run to compare different fine-tuning strategies:

| Experiment       | Trainable Params | Accuracy | F1 Score |
|------------------|-----------------|----------|----------|
| Frozen BERT      | ~7.7K (head only) | 38.1%  | 0.27     |
| Full Fine-Tuning | ~110M (all layers) | 55.1%  | 0.48     |
| Last 2 Layers    | ~14M              | 55.6%  | 0.50 ✅  |

> 10-class classification problem trained on limited compute.  
> Random baseline would be ~10%, so these results reflect genuine learning.

### Key Observation
Fine-tuning only the last 2 BERT encoder layers gave the best results — matching full fine-tuning performance with far fewer parameters updated.

---

## Evaluation Metrics
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)
- Confusion Matrix

---

## How to Run

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rmisra/news-category-dataset) and place `news.json` in the project directory
2. Open `Task4_BERT_FineTuning.ipynb` in Google Colab or Jupyter Notebook
3. Enable GPU runtime (recommended): Runtime → Change runtime type → T4 GPU
4. Run all cells top to bottom

---

## File Structure

```
├── Task4_BERT_FineTuning.ipynb   # Main notebook
├── news.json                      # Dataset (download from Kaggle)
└── README.md                      # This file
```

---

## Note
I understood the pipeline end-to-end up to model building.  
The experiments section was done with AI assistance — I'm still learning and this project is part of that process.
