import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
import numpy as np
from pathlib import Path
from typing import List


class ImageEmbeddingSystem:
    def __init__(self):
        # Initialize CLIP model
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Connect to Milvus
        self.setup_milvus()

    def setup_milvus(self):
        try:
            # Connect to Milvus server
            connections.connect(host='localhost', port='19530')

            # Define collection schema
            dim = 512  # CLIP embedding dimension
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            schema = CollectionSchema(fields=fields, description="image_embeddings")

            # Create collection
            self.collection = Collection(name="image_embeddings", schema=schema)

            # Create index
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)

        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}")

    def generate_embedding(self, image_path: Path) -> np.ndarray:
        try:
            # Load and preprocess image
            with Image.open(image_path) as image:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            # Generate embedding
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)

            # Convert to numpy and normalize
            embedding = image_features.cpu().numpy()[0]
            embedding = embedding / np.linalg.norm(embedding)
            return embedding

        except Exception as e:
            raise ValueError(f"Failed to generate embedding for {image_path}: {e}")

    def check_milvus_entries(self):
        """Check and display information about stored embeddings"""
        try:
            # Get collection info
            self.collection.load()
            entity_count = self.collection.num_entities

            print(f"Total number of stored embeddings: {entity_count}")

            if entity_count > 0:
                # Retrieve some samples to verify content
                results = self.collection.query(
                    expr="id >= 0",
                    output_fields=["id", "image_path"],
                    limit=5  # Show first 5 entries as sample
                )

                print("\nSample entries:")
                for entry in results:
                    print(f"ID: {entry['id']}, Path: {entry['image_path']}")

                # Verify if embeddings exist
                sample = self.collection.query(
                    expr="id >= 0",
                    output_fields=["embedding"],
                    limit=1
                )
                if sample and len(sample[0]['embedding']) == 512:  # CLIP dimension
                    print("\nEmbeddings verified: ✓ (correct dimension of 512)")
                else:
                    print("\nWarning: Embeddings may not be stored correctly")

        except Exception as e:
            print(f"Error checking Milvus entries: {e}")
        finally:
            self.collection.release()  # Release collection from memory


    def process_and_store_images(self, image_paths: List[Path]):
        if not image_paths:
            raise ValueError("No image paths provided")

        # Prepare data for batch insertion
        embeddings = []
        paths = []

        # Process images in batches
        batch_size = 100  # Adjust based on your system's capacity

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            for image_path in batch_paths:
                try:
                    embedding = self.generate_embedding(image_path)
                    embeddings.append(embedding)
                    paths.append(str(image_path))
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    continue

            # Insert batch into Milvus
            if embeddings and paths:
                try:
                    self.collection.insert([paths, embeddings])
                    self.collection.flush()
                    print(f"Successfully processed batch of {len(paths)} images")
                    embeddings = []
                    paths = []
                except Exception as e:
                    print(f"Error inserting batch into Milvus: {e}")

