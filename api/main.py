# api/main.py
"""
Camada HTTP fina sobre o ScientificAVRPipeline, para permitir que uma
aplicacao web (ou qualquer cliente HTTP) rode a analise de AVR sem precisar
importar o codebase Python diretamente.

Uso local:
    pip install -r requirements-api.txt   # alem de requirements.txt ou requirements-rocm.txt
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Teste rapido:
    curl -F "image=@data/DRIVE/test/images/01_test.tif" http://localhost:8000/analyze

Sem autenticacao, sem deploy em produção -- serve para integracao local/
desenvolvimento de uma aplicacao web em cima do pipeline existente.
"""

import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from src.pipeline.integrated_pipeline import ScientificAVRPipeline

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

_pipeline: ScientificAVRPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Carregando ScientificAVRPipeline...")
    _pipeline = ScientificAVRPipeline()
    if not _pipeline.load_models():
        # Nao derruba o processo -- /health e /analyze reportam o problema
        # de forma clara em vez do servidor simplesmente nao subir.
        logger.error(
            "Falha ao carregar os modelos do pipeline -- /analyze vai falhar "
            "ate que checkpoints validos existam em models/segmentation e "
            "models/av_classification."
        )
    yield
    _pipeline = None


app = FastAPI(
    title="Retinal AVR Pipeline API",
    description="Analise automatica de AVR (arteriolo-venular ratio) a partir de retinografias.",
    version="0.1.0",
    lifespan=lifespan,
)


def _json_safe(value):
    """Converte tipos numpy/torch para tipos nativos serializaveis em JSON."""
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return None  # mascaras/arrays nao vao na resposta JSON (ver /analyze)
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
            content={"status": "unavailable", "detail": "Pipeline nao inicializado (checkpoints ausentes?)."},
        )
    return {"status": "ok", "device": str(_pipeline.device)}


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    """
    Recebe uma imagem de retinografia e roda o pipeline completo:
    segmentacao de vasos -> classificacao A/V -> deteccao do disco optico
    -> calculo cientifico do AVR (Zona B, Knudtson) -> risco cardiovascular.
    """
    if _pipeline is None or not _pipeline.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Pipeline nao inicializado -- verifique se ha checkpoints treinados em models/.",
        )

    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Extensao '{suffix}' nao suportada. Use uma de: {sorted(ALLOWED_SUFFIXES)}",
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(contents)
        tmp.flush()

        results = _pipeline.process_image(tmp.name, verbose=False)

    if not results or results.get("error"):
        raise HTTPException(
            status_code=422,
            detail=f"Falha ao processar a imagem: {results.get('error') if results else 'sem resultado'}",
        )

    # Remove arrays/tensors grandes (mask, predictions) da resposta -- a API
    # devolve metricas e metadados, nao as mascaras binarias completas.
    excluded = {"mask", "predictions"}
    response = {k: _json_safe(v) for k, v in results.items() if k not in excluded}
    return response
