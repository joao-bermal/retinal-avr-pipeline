# api/main.py
"""
Thin HTTP layer over ScientificAVRPipeline, so a web application (or any
HTTP client) can run the AVR analysis without importing the Python codebase
directly.

Local usage:
    pip install -r requirements-api.txt   # on top of requirements.txt or requirements-rocm.txt
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Quick test:
    curl -F "image=@data/DRIVE/test/images/01_test.tif" http://localhost:8000/analyze

No authentication, no production deployment story -- this is for local
integration/development of a web application on top of the existing
pipeline (see frontend/ for the Next.js UI that calls this).
"""

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.pipeline.integrated_pipeline import ScientificAVRPipeline

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_pipeline: ScientificAVRPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Loading ScientificAVRPipeline...")
    _pipeline = ScientificAVRPipeline()
    if not _pipeline.load_models():
        # Don't crash the process -- /health and /analyze report the
        # problem clearly instead of the server simply failing to start.
        logger.error(
            "Failed to load pipeline models -- /analyze will fail until "
            "valid checkpoints exist under models/segmentation and "
            "models/av_classification."
        )
    yield
    _pipeline = None


app = FastAPI(
    title="Retinal AVR Pipeline API",
    description="Automatic AVR (arteriolar-to-venular ratio) analysis from fundus photographs.",
    version="0.1.0",
    lifespan=lifespan,
)

# Allow the local Next.js dev server (and any other local frontend) to call
# this API directly from the browser. Not meant for a public deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _json_safe(value):
    """Convert numpy/torch types to native JSON-serializable types."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return None  # masks/arrays don't go in the JSON response (see /analyze)
    if isinstance(value, torch.Tensor):
        return None
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


@app.get("/health")
def health():
    if _pipeline is None or not _pipeline.is_initialized:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "Pipeline not initialized (missing checkpoints?)."},
        )
    return {"status": "ok", "device": str(_pipeline.device)}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    """
    Accepts a fundus image and runs the full pipeline: vessel segmentation ->
    A/V classification -> optic disc detection -> scientific AVR calculation
    (Zone B, Knudtson) -> cardiovascular risk.
    """
    if _pipeline is None or not _pipeline.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not initialized -- check that trained checkpoints exist under models/.",
        )

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported extension '{suffix}'. Use one of: {sorted(ALLOWED_SUFFIXES)}",
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()

        results = _pipeline.process_image(tmp.name, verbose=False)

    if not results or results.get("error"):
        raise HTTPException(
            status_code=422,
            detail=f"Failed to process image: {results.get('error') if results else 'no result'}",
        )

    # Drop large arrays/tensors (mask, predictions) from the response -- the
    # API returns metrics and metadata, not the full binary masks.
    excluded = {"mask", "predictions"}
    response = {k: _json_safe(v) for k, v in results.items() if k not in excluded}
    return response
