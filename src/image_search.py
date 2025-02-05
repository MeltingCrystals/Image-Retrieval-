import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import Collection
import numpy as np


class TextImageSearcher:
    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.collection = Collection("image_embeddings")

    def generate_text_embedding(self, text: str) -> np.ndarray:
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        embedding = text_features.cpu().numpy()[0]
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def search(self, text_query: str, top_k: int = 5):
        # Generate embedding for text query
        text_embedding = self.generate_text_embedding(text_query)

        # Search in Milvus
        self.collection.load()
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[text_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k * 2,
            output_fields=["image_path"]
        )

        # Deduplicate results
        seen_paths = set()
        unique_matches = []

        for hits in results:
            for hit in hits:
                path = hit.entity.get('image_path')
                if path not in seen_paths and hit.score > 0.25:  # Filter low scores
                    seen_paths.add(path)
                    unique_matches.append({
                        'path': path,
                        'score': hit.score
                    })
                if len(unique_matches) >= top_k:
                    break

        return unique_matches[:top_k]