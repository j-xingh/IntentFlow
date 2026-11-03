from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

MODEL_DIR = Path(
    os.getenv(
        "INTENTFLOW_MODEL_DIR",
        ROOT_DIR / "models" / "distilbert" / "final_fp32",
    )
)

MAPPING_PATH = Path(
    os.getenv(
        "INTENTFLOW_LABEL_MAPPING",
        ROOT_DIR / "models" / "distilbert" / "label_mapping.json",
    )
)

MAX_LENGTH = 128
MAX_TEXT_LENGTH = 1000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("intentflow")

tokenizer = None
model = None

id_to_label: dict[int, str] = {}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# MODEL LIFECYCLE
# ---------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model, id_to_label

    logger.info("Loading IntentFlow model...")

    if not MODEL_DIR.is_dir():
        raise RuntimeError(
            f"Model directory not found: {MODEL_DIR}. "
            "Restore the fine-tuned model files before starting the API."
        )

    if not MAPPING_PATH.is_file():
        raise RuntimeError(
            f"Label mapping not found: {MAPPING_PATH}"
        )

    # Load label mapping
    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping: dict[str, Any] = json.load(file)

    raw_mapping = mapping.get("id_to_label")

    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise RuntimeError(
            "Expected label_mapping.json to contain "
            "a non-empty 'id_to_label' object."
        )

    id_to_label = {
        int(label_id): str(intent)
        for label_id, intent in raw_mapping.items()
    }

    # Load tokenizer and fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.to(device)
    model.eval()

    logger.info("Model loaded successfully.")
    logger.info("Device: %s", device)
    logger.info("Number of intent classes: %s", len(id_to_label))

    yield

    # Cleanup
    model = None
    tokenizer = None

    logger.info("IntentFlow model unloaded.")


# ---------------------------------------------------------
# FASTAPI APPLICATION
# ---------------------------------------------------------

app = FastAPI(
    title="IntentFlow API",
    description="Predict a banking intent from a short customer message.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------------

allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "INTENTFLOW_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# ---------------------------------------------------------

class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
    )


class Prediction(BaseModel):
    intent: str
    confidence: float


class PredictResponse(BaseModel):
    text: str
    prediction: Prediction
    top_predictions: list[Prediction]


# ---------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------

@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "IntentFlow API",
        "status": "running",
        "docs": "/docs",
    }


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": model is not None and tokenizer is not None,
        "device": str(device),
    }


# ---------------------------------------------------------
# PREDICTION ENDPOINT
# ---------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(payload: PredictRequest) -> PredictResponse:

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="The classification model is not loaded.",
        )

    text = payload.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text must not be blank.",
        )

    try:
        # Tokenize input
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            return_token_type_ids=False,
        )

        # Defensive removal:
        # DistilBERT does not accept token_type_ids.
        encoded.pop("token_type_ids", None)

        # Move tensors to the selected device
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        # Model inference
        with torch.inference_mode():

            outputs = model(**encoded)

            logits = outputs.logits[0]

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            count = min(
                payload.top_k,
                probabilities.shape[-1],
            )

            scores, indices = torch.topk(
                probabilities,
                k=count,
            )

        # Format predictions
        top_predictions = [
            Prediction(
                intent=id_to_label.get(
                    int(index),
                    f"label_{int(index)}",
                ),
                confidence=float(score),
            )
            for score, index in zip(
                scores.cpu().tolist(),
                indices.cpu().tolist(),
            )
        ]

        # Return response
        return PredictResponse(
            text=text,
            prediction=top_predictions[0],
            top_predictions=top_predictions,
        )

    except HTTPException:
        raise

    except Exception as error:

        # Keep the complete exception in the backend terminal.
        logger.exception(
            "Prediction failed for input: %s",
            text,
        )

        # Do not expose internal implementation details.
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Please try again.",
        ) from error