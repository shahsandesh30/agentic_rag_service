from __future__ import annotations
from typing import List, Literal
import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Thin wrapper so we can swap models later.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str|None = None, normalize: bool = True):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device or ("cuda" if _has_cuda() else "cpu"))
        self.normalize = normalize
        #cache dim
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )

        # ensure contiguous float32 for storage
        arr = np.asarray(vecs, dtype=np.float32)
        return arr
    
def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    
    except Exception:
        return False