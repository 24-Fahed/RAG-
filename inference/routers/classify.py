"""分类推理端点。

封装 rag_langgraph.models.classifier.RetrievalClassifier。
"""

from fastapi import APIRouter
from pydantic import BaseModel

from rag_langgraph.models.classifier import get_classifier
from inference.config import CLASSIFICATION_MODEL

router = APIRouter()


class ClassifyRequest(BaseModel):
    query: str
    model_name: str = CLASSIFICATION_MODEL
    weights_path: str | None = None


class ClassifyResponse(BaseModel):
    label: int  # 0 = 无需检索, 1 = 需要检索


@router.post("/classify", response_model=ClassifyResponse)
def classify(req: ClassifyRequest):
    classifier = get_classifier(
        model_name=req.model_name,
        weights_path=req.weights_path,
    )
    label = classifier.predict(req.query)
    return ClassifyResponse(label=label)
