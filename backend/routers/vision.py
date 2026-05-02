"""
routers/vision.py — POST /api/classify endpoint.

Accepts a leaf image (multipart/form-data), runs CNN inference,
returns disease class + confidence + Arabic disease info.
"""

import io
import logging

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from typing import Optional, Dict
from pydantic import BaseModel
from PIL import Image

router = APIRouter()
logger = logging.getLogger(__name__)


class ClassifyResponse(BaseModel):
    class_id:    int
    class_name:  str
    class_ar:    str
    class_fr:    str
    confidence:  float
    low_conf:    bool
    eppo_code:   Optional[str]
    advice_ar:   str
    all_scores:  Dict[str, float]


@router.post("/classify", response_model=ClassifyResponse)
async def classify_leaf(
    request: Request,
    file: UploadFile = File(..., description="Leaf image (jpg/png/webp)")
):
    """
    Classify olive leaf disease from uploaded image.
    Returns disease class, confidence, and Arabic advice.
    """
    state = request.state.app_state
    cnn = state.get("cnn")

    if cnn is None:
        raise HTTPException(
            503,
            "CNN model not loaded — train the model first (see vision/train.py)"
        )

    # Read image bytes
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(400, "Empty image file")
    if len(image_bytes) > 10 * 1024 * 1024:   # 10MB limit
        raise HTTPException(413, "Image too large (max 10MB)")

    # Validate image
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.verify()   # check it's a valid image
        image = Image.open(io.BytesIO(image_bytes))  # re-open after verify
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    # Run CNN inference
    result = cnn.predict(image)

    # Enrich with Arabic disease info
    from config import DISEASE_CLASSES
    disease = DISEASE_CLASSES.get(result["class_id"], {})

    logger.info(
        f"CNN: class={result['class_name']} "
        f"conf={result['confidence']:.3f} "
        f"low_conf={result['low_conf']}"
    )

    return ClassifyResponse(
        class_id   = result["class_id"],
        class_name = result["class_name"],
        class_ar   = disease.get("ar", "غير معروف"),
        class_fr   = disease.get("fr", "Inconnu"),
        confidence = round(result["confidence"], 4),
        low_conf   = result["low_conf"],
        eppo_code  = disease.get("eppo"),
        advice_ar  = disease.get("advice_ar", ""),
        all_scores = {k: round(v, 4) for k, v in result["all_scores"].items()},
    )