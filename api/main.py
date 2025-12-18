from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from retrieval.search import search
from reranking.reranker import rerank
from api.formatter import format_assessment
from api.schemas import RecommendResponse

import json
from pathlib import Path
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


app = FastAPI(title="SHL Recommendation API")


# Load canonical data once
DATA_FILE = Path("data/processed/shl_assessments.json")
ASSESSMENTS = {
    a["assessment_id"]: a for a in json.loads(DATA_FILE.read_text(encoding="utf-8"))
}


class RecommendRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Phase-2
    retrieved = search(query)[:50]

    # Attach metadata
    candidates = []
    for r in retrieved:
        aid = r.get("assessment_id")
        if aid in ASSESSMENTS:
            c = ASSESSMENTS[aid].copy()
            c["retrieval_score"] = r.get("retrieval_score", 0)
            candidates.append(c)

    if not candidates:
        raise HTTPException(status_code=404, detail="No recommendations found")

    # Phase-3
    reranked = rerank(query, candidates, final_k=10)

    formatted = [format_assessment(a) for a in reranked]

    return {"recommended_assessments": formatted}
