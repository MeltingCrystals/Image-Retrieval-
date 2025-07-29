# ImageEmbeddingSystem.py
"""Module for generating and storing image embeddings in Milvus."""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import torch
from PIL import Image
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from config import MILVUS_HOST, MILVUS_PORT, EMBEDDING_DIM, BATCH_SIZE

logger = logging.getLogger(__name__)


class ImageEmbeddingSystem:
    """Handles image embedding generation and storage in Milvus."""

    def __init__(self, model: CLIPModel, processor: CLIPProcessor, device: str):
        """
        Initializes the ImageEmbeddingSystem.

        Args:
            model: The CLIP model instance.
            processor: The CLIP processor instance.
            device: The device to use ('cuda' or 'cpu').
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.setup_milvus()

    def setup_milvus(self):
        """Sets up the Milvus connection and collection."""
        try:
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            logger.info("Connected to Milvus server.")

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
                # Add field to store the original unnormalized embedding magnitude
                FieldSchema(name="magnitude", dtype=DataType.FLOAT)
            ]
            schema = CollectionSchema(fields=fields, description="Image embeddings collection")
            collection_name = "image_embeddings"

            if utility.has_collection(collection_name):
                logger.info(f"Collection '{collection_name}' already exists.")
                self.collection = Collection(collection_name)
            else:
                self.collection = Collection(name=collection_name, schema=schema)
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
                logger.info(f"Created new collection '{collection_name}' with index.")

        except Exception as e:
            logger.error(f"Failed to set up Milvus: {e}")
            raise

    def generate_embedding(self, image_path: Path) -> Tuple[np.ndarray, float]:
        """
        Generates an embedding for an image and returns both normalized embedding and magnitude.

        Args:
            image_path: Path to the image file.

        Returns:
            Tuple of (normalized_embedding, magnitude)

        Raises:
            Exception: If embedding generation fails.
        """
        try:
            with Image.open(image_path) as image:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Get unnormalized embedding
            embedding = image_features.cpu().numpy()[0]

            # Calculate magnitude (L2 norm)
            magnitude = float(np.linalg.norm(embedding))

            # Return normalized embedding and its magnitude
            return embedding / magnitude, magnitude

        except Exception as e:
            logger.error(f"Failed to generate embedding for {image_path}: {e}")
            raise

    def process_and_store_images(self, image_paths: List[Path]) -> Tuple[int, int]:
        """
        Processes images in batches and stores their embeddings in Milvus.

        Args:
            image_paths: List of image file paths.

        Returns:
            Tuple of (successful_count, failed_count)
        """
        if not image_paths:
            logger.warning("No image paths provided for processing.")
            return 0, 0

        normalized_embeddings = []
        magnitudes = []
        paths = []
        failed_count = 0

        # Process images with progress bar
        for image_path in tqdm(image_paths, desc="Generating embeddings"):
            try:
                normalized_embedding, magnitude = self.generate_embedding(image_path)
                normalized_embeddings.append(normalized_embedding)
                magnitudes.append(magnitude)
                paths.append(str(image_path))
            except Exception as e:
                logger.warning(f"Skipping {image_path} due to error: {e}")
                failed_count += 1
                continue

        successful_count = len(paths)

        if normalized_embeddings and paths:
            try:
                # Now we store both normalized embeddings and magnitudes
                self.collection.insert([paths, normalized_embeddings, magnitudes])
                self.collection.flush()
                logger.info(f"Inserted batch of {len(paths)} images into Milvus.")
            except Exception as e:
                logger.error(f"Error inserting batch into Milvus: {e}")
                # If the entire batch failed, count all as failures
                failed_count += successful_count
                successful_count = 0

        return successful_count, failed_count

    def get_embeddings(self, limit: int = 1000) -> List[Tuple[str, np.ndarray]]:
        """
        Retrieves normalized embeddings and image paths from Milvus.

        Args:
            limit: Maximum number of embeddings to retrieve.

        Returns:
            List of tuples (image_path, normalized_embedding).
        """
        try:
            self.collection.load()
            results = self.collection.query(
                expr="id >= 0",
                output_fields=["image_path", "embedding"],
                limit=limit
            )
            embeddings = [(entry["image_path"], np.array(entry["embedding"])) for entry in results]
            logger.info(f"Retrieved {len(embeddings)} embeddings from Milvus.")
            return embeddings
        except Exception as e:
            logger.error(f"Error retrieving embeddings from Milvus: {e}")
            raise
        finally:
            self.collection.release()

    def get_embeddings_with_magnitude(self, limit: int = 1000) -> List[Tuple[str, np.ndarray, float]]:
        """
        Retrieves normalized embeddings, magnitudes, and image paths from Milvus.

        Args:
            limit: Maximum number of embeddings to retrieve.

        Returns:
            List of tuples (image_path, normalized_embedding, magnitude).
        """
        try:
            self.collection.load()
            results = self.collection.query(
                expr="id >= 0",
                output_fields=["image_path", "embedding", "magnitude"],
                limit=limit
            )
            embeddings = [
                (entry["image_path"],
                 np.array(entry["embedding"]),
                 entry.get("magnitude", 1.0))  # Default to 1.0 if magnitude not available
                for entry in results
            ]
            logger.info(f"Retrieved {len(embeddings)} embeddings with magnitudes from Milvus.")
            return embeddings
        except Exception as e:
            logger.error(f"Error retrieving embeddings with magnitudes from Milvus: {e}")
            raise
        finally:
            self.collection.release()

    def reconstruct_original_embeddings(self, embeddings: List[Tuple[str, np.ndarray, float]]) -> List[
        Tuple[str, np.ndarray]]:
        """
        Reconstructs original unnormalized embeddings from normalized embeddings and magnitudes.

        Args:
            embeddings: List of tuples (image_path, normalized_embedding, magnitude).

        Returns:
            List of tuples (image_path, unnormalized_embedding).
        """
        return [(path, emb * mag) for path, emb, mag in embeddings]