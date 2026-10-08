import numpy as np
from task2_bot.config import MODEL

class Embedder:
    """Multilingual E5: prefixes are required by the model card."""
    def __init__(self, model=MODEL, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.encoder = SentenceTransformer(model, device=device, trust_remote_code=False)
        self.tokenizer = self.encoder.tokenizer

    def passages(self, texts):
        return np.asarray(self.encoder.encode(["passage: " + t for t in texts], batch_size=16,
                          normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)

    def query(self, text):
        return np.asarray(self.encoder.encode(["query: " + text], normalize_embeddings=True,
                          show_progress_bar=False)[0], dtype=np.float32)
