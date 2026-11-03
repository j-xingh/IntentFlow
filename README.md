# IntentFlow

**Understanding banking customer intent through machine learning and transformer-based NLP.**

IntentFlow is a multi-class text classification system that identifies the intent behind short banking-related customer messages. It compares a traditional TF-IDF and Logistic Regression pipeline against fine-tuned DistilBERT models, with an interactive React interface powered by a FastAPI inference backend.

The system classifies messages into **77 banking intent categories** using the Banking77 dataset.

---

## Overview

Customer messages often express similar problems using different language. IntentFlow explores how classical machine learning and transformer-based language models can distinguish between these intents.

The project brings together:

- A reproducible text classification workflow.
- A classical machine learning baseline.
- Fine-tuned DistilBERT models.
- Model evaluation and error analysis.
- A FastAPI inference service.
- An interactive React frontend.

## Model Performance

The following results were obtained on the **1,501-example validation set**.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 87.74% | **87.87%** | 87.73% |
| DistilBERT FP16 | 87.94% | 87.02% | 87.78% |
| DistilBERT FP32 | **88.27%** | 87.40% | **88.15%** |

### Key findings

- DistilBERT FP32 achieved the highest validation accuracy and weighted F1.
- Logistic Regression achieved the highest macro F1.
- The classical baseline remains competitive despite its substantially simpler architecture.
- Aggregate metrics do not capture every class-level weakness. Error analysis revealed differences in how the models handled individual intents.

The official test set has not yet been used for a reported final evaluation. All results above are validation results, not test results.

## Dataset

IntentFlow uses **Banking77**, a dataset containing short banking customer-service queries classified into 77 distinct intents.

| Split | Examples | Purpose |
|---|---:|---|
| Training | 8,502 | Model training |
| Validation | 1,501 | Model comparison and selection |
| Test | 3,080 | Held-out final evaluation |

The official test split is reserved for final evaluation.

Dataset redistribution and attribution requirements should be followed when obtaining or using Banking77. The raw and processed datasets are excluded from this repository.

## Models

### 1. TF-IDF + Logistic Regression

The baseline converts customer messages into TF-IDF feature vectors and uses Logistic Regression to predict the corresponding intent.

This provides a computationally efficient benchmark against which the transformer models can be evaluated.

### 2. Fine-tuned DistilBERT

The transformer model uses `distilbert-base-uncased`, fine-tuned for the 77-class classification task.

Recorded training configuration:

| Parameter | Configuration |
|---|---|
| Base model | DistilBERT |
| Maximum sequence length | 128 |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Per-device batch size | 4 |
| Gradient accumulation | 2 |
| Effective batch size | 8 |
| Padding | Dynamic |

The application currently uses the FP32 model.

**Model artifacts are not stored in this GitHub repository.** The trained weights and tokenizer must be obtained separately before running inference. The label mapping is stored at:

```text
models/distilbert/label_mapping.json
```

Expected model directory:

```text
models/distilbert/final_fp32/
```

## Application Architecture

```text
                    ┌────────────────────────┐
                    │      React + Vite      │
                    │       Frontend         │
                    └───────────┬────────────┘
                                │
                                │ HTTP Request
                                ▼
                    ┌────────────────────────┐
                    │        FastAPI         │
                    │     Inference API      │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │       DistilBERT       │
                    │    Fine-tuned FP32     │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Predicted Intent     │
                    │   Top-K Predictions    │
                    └────────────────────────┘
```

The frontend sends customer messages to the FastAPI backend. The backend tokenizes the input, runs model inference, and returns the predicted intent along with the top-ranked alternatives.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python, JavaScript |
| Machine Learning | Scikit-learn |
| Deep Learning | PyTorch |
| NLP | Hugging Face Transformers |
| Feature Extraction | TF-IDF |
| Backend | FastAPI, Uvicorn |
| Frontend | React, Vite |
| Model | DistilBERT |
| Dataset | Banking77 |

## Project Structure

```text
IntentFlow/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── styles.css
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── privacy.html
│   └── terms.html
│
├── models/
│   └── distilbert/
│       └── label_mapping.json
│
├── reports/
│
├── src/
│   ├── analyze_errors.py
│   ├── compare_models.py
│   ├── diagnose_data.py
│   ├── eda.py
│   ├── error_analysis.py
│   ├── prepare_data.py
│   ├── prepare_transformer_data.py
│   ├── train_baseline.py
│   └── train_distilbert.py
│
├── tests/
├── .gitignore
├── requirements-app.txt
└── README.md
```

Local datasets, virtual environments, model weights, training checkpoints, logs, and temporary artifacts are excluded from version control.

## Getting Started

### Prerequisites

- Python 3.11
- Node.js and npm
- Git
- Trained DistilBERT model artifacts

### 1. Clone the repository

```bash
git clone https://github.com/j-xingh/IntentFlow.git
cd IntentFlow
```

### 2. Set up the Python environment

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the application dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements-app.txt
```

PyTorch installation may need to be adjusted depending on whether inference will run on CPU or a CUDA-enabled GPU.

### 3. Restore the model artifacts

Place the fine-tuned DistilBERT model files in:

```text
models/distilbert/final_fp32/
```

Ensure that the label mapping exists at:

```text
models/distilbert/label_mapping.json
```

The model directory must contain its model configuration, tokenizer files, and trained model weights.

The application cannot perform inference without these artifacts.

### 4. Start the backend

From the repository root:

```bash
uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 5. Set up the frontend

Open a second terminal:

```powershell
cd frontend
npm install
```

Configure the frontend API URL using the existing `.env.example` as a reference. For local development, the API base URL should point to:

```text
http://127.0.0.1:8000
```

Start the development server:

```bash
npm run dev
```

Vite will display the local frontend URL, normally:

```text
http://localhost:5173
```

Keep both the frontend and backend running while using the application.

## API Reference

### `GET /health`

Checks the backend health and whether the model has been loaded.

### `POST /predict`

Accepts a customer message and returns its predicted intent.

**Request:**

```json
{
  "text": "My virtual card is not working.",
  "top_k": 3
}
```

**Response:**

```json
{
  "text": "My virtual card is not working.",
  "prediction": {
    "intent": "getting_virtual_card",
    "confidence": 0.4646
  },
  "top_predictions": [
    {
      "intent": "getting_virtual_card",
      "confidence": 0.4646
    },
    {
      "intent": "virtual_card_not_working",
      "confidence": 0.0826
    },
    {
      "intent": "card_not_working",
      "confidence": 0.0317
    }
  ]
}
```

The response illustrates a recorded local inference example. Confidence values are model softmax scores and are not calibrated probabilities or guarantees of correctness.

## Deployment

The intended deployment architecture separates the frontend, backend, and trained model artifacts:

| Component | Platform | Responsibility |
|---|---|---|
| Frontend | Vercel | React application |
| Backend | Render | FastAPI inference service |
| Model artifacts | Hugging Face Hub | Fine-tuned DistilBERT weights |
| Source code | GitHub | Application and training code |

The trained model is kept separate from the source repository because its individual weight files exceed GitHub's standard file-size limit.

Production deployment requires configuring the backend to retrieve the model artifacts, setting the frontend API URL, and configuring backend CORS to allow the deployed frontend origin.

## Evaluation and Analysis

The project includes scripts for:

- Exploratory data analysis.
- Dataset preparation and preprocessing.
- Baseline model training.
- Transformer fine-tuning.
- Model comparison.
- Prediction error analysis.
- Per-intent performance inspection.

The `reports/` directory contains selected evaluation outputs, including model comparison metrics and error-analysis reports.

## Limitations

- Reported model performance is based on the validation split.
- Final evaluation on the untouched official test set remains outstanding.
- Model performance varies across individual intent categories.
- Softmax confidence scores are not calibrated.
- Inference requires the separately hosted or locally restored model artifacts.
- The current application is not intended to replace official banking customer-support systems.

## License and Attribution

Banking77 and the pretrained DistilBERT model are subject to their respective licensing and attribution requirements.

A project-level license should be added before redistributing this repository. Dataset, pretrained model, fine-tuned weights, and project source code licensing should be considered separately.