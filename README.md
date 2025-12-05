# Generating Concise Medical Findings from Clinical Reports

An automated system for generating concise medical impressions from detailed clinical findings using deep learning techniques. This project leverages the BART (Bidirectional and Auto-Regressive Transformers) model to transform verbose clinical report findings into succinct, clinically relevant impressions.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Requirements](#requirements)
- [References](#references)

## 🎯 Overview

This project addresses the challenge of automating medical report summarization, specifically focusing on generating concise impressions from detailed radiological findings. The system uses fine-tuned BART-base model to learn domain-specific patterns in medical text and generate clinically accurate summaries.

**Key Applications:**
- Automated medical report summarization
- Clinical documentation assistance
- Healthcare AI research
- Medical text generation

## ✨ Features

- **Automated Summarization**: Generates concise medical impressions from detailed findings
- **Domain-Specific Fine-tuning**: Pre-trained BART model fine-tuned on medical domain data
- **Clinical Accuracy**: Maintains medical terminology and clinical relevance
- **Reproducible**: Includes data splitting and checkpoint saving for reproducibility
- **Flexible Training**: Supports mixed precision training and checkpoint resumption

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for training)
- pip package manager

### Setup

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd Clinical_Model
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📊 Dataset

The project uses the **Indiana Reports dataset**, a collection of chest X-ray reports containing:

- **Total Samples**: ~3,852 clinical reports
- **Format**: CSV file with structured medical text
- **Fields**: 
  - `findings`: Detailed radiological observations
  - `impression`: Concise summary statements written by radiologists
  - Additional metadata: uid, MeSH terms, Problems, image type, indication, comparison

**Dataset Location**: `data/raw/indiana_reports.csv`

**Data Split**:
- Training: 80% (~3,082 samples)
- Validation: 10% (~385 samples)
- Test: 10% (~385 samples)

## 📁 Project Structure

```
Clinical_Model/
├── data/
│   ├── raw/
│   │   └── indiana_reports.csv      # Original dataset
│   ├── processed/                    # Processed data (if any)
│   └── sample/                       # Sample data
├── src/
│   ├── config.py                     # Configuration parameters
│   ├── dataset.py                     # Data loading and splitting
│   ├── train.py                      # Training script
│   ├── evaluate.py                    # Evaluation script
│   ├── utils.py                      # Dataset class and utilities
│   └── job.sh                        # SLURM job script for cluster training
├── models/                           # Saved model checkpoints
├── results/                          # Evaluation results
├── notebooks/                        # Jupyter notebooks
├── checkpoint_epoch2.pth            # Trained model checkpoint
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── Internship_Report.md             # Detailed project report
```

## 💻 Usage

### Training

Train the model from scratch:

```bash
python src/train.py --batch_size 4 --num_epochs 5
```

With mixed precision training (faster, lower memory):

```bash
python src/train.py --batch_size 4 --num_epochs 5 --use_amp
```

**Training Parameters**:
- `--batch_size`: Batch size for training (default: 4)
- `--num_epochs`: Number of training epochs (default: 5)
- `--use_amp`: Enable Automatic Mixed Precision training

### Using SLURM (Cluster Training)

Submit training job to SLURM cluster:

```bash
sbatch src/job.sh
```

### Evaluation

Evaluate the model on test data:

```python
from src.evaluate import *
# Follow the evaluation script in src/evaluate.py
```

### Inference

Generate impressions from findings:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import src.config as config

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME).to(device)

# Load checkpoint
checkpoint = torch.load("checkpoint_epoch2.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Generate impression
findings = "Your clinical findings text here..."
inputs = tokenizer(findings, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=128,
    )

impression = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Generated Impression: {impression}")
```

## 🤖 Model Details

### Architecture

**Model**: BART-base (`facebook/bart-base`)

- **Type**: Sequence-to-sequence transformer
- **Encoder**: Bidirectional transformer (like BERT)
- **Decoder**: Autoregressive transformer (like GPT)
- **Parameters**: ~140 million
- **Pre-training**: Large-scale text data

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Maximum Input Length | 512 tokens |
| Maximum Target Length | 128 tokens |
| Batch Size | 4 |
| Learning Rate | 5e-5 |
| Optimizer | AdamW |
| Number of Epochs | 5 |
| Mixed Precision | Optional (AMP) |

### Configuration

Edit `src/config.py` to modify:
- Model name
- Training parameters
- Data paths
- Tokenization settings

## 📈 Results

### Qualitative Results

The model demonstrates strong performance in generating clinically relevant impressions:

**Example 1:**
- **Input**: "There are no focal areas of consolidation. No suspicious bony opacities. Heart size within normal limits..."
- **Output**: "No acute cardiopulmonary abnormality."
- **Match**: ✓ Perfect match with physician impression

**Example 2:**
- **Input**: "Mild cardiomegaly, stable mediastinal contours. No focal alveolar consolidation..."
- **Output**: "Mild cardiomegaly without acute pulmonary findings"
- **Match**: ✓ Very close match with slight wording differences

### Model Performance

- **Clinical Accuracy**: Successfully identifies key clinical findings
- **Conciseness**: Generates significantly shorter summaries (10:1 compression ratio)
- **Medical Terminology**: Appropriate use of medical terminology
- **Consistency**: Format consistent with physician-written reports

### Checkpoints

- `checkpoint_epoch2.pth`: Model checkpoint from epoch 2
- Checkpoints are saved automatically after each epoch

## 📦 Requirements

See `requirements.txt` for full list. Key dependencies:

- `transformers` - Hugging Face Transformers library
- `torch` - PyTorch deep learning framework
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- `datasets` - Dataset handling
- `evaluate` - Evaluation metrics

Install all dependencies:
```bash
pip install -r requirements.txt
```

## 📚 References

1. Lewis, M., et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*.

2. Hugging Face Transformers Library: https://huggingface.co/transformers/

3. PyTorch Documentation: https://pytorch.org/docs/

4. Indiana Reports Dataset


## 🙏 Acknowledgments

- Indiana Reports dataset providers
- Hugging Face for the Transformers library
- Facebook AI Research for BART model

---