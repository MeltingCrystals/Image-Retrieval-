# enhanced_image_search.py
"""Enhanced image search module using geometric similarity metrics."""

import logging
import numpy as np
import torch
from pymilvus import Collection
from transformers import CLIPProcessor, CLIPModel
from config import SCORE_THRESHOLD
from geometric_metrics import GeometricSimilarityMetrics

logger = logging.getLogger(__name__)


class EnhancedTextImageSearcher:
    """Handles text-based image search using multiple geometric similarity metrics."""

    def __init__(self, model: CLIPModel, processor: CLIPProcessor, device: str):
        """
        Initializes the EnhancedTextImageSearcher.

        Args:
            model: The CLIP model instance.
            processor: The CLIP processor instance.
            device: The device to use ('cuda' or 'cpu').
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.collection = Collection("image_embeddings")
        self.metrics = GeometricSimilarityMetrics()

        # Default parameters for optimized similarity
        self.similarity_params = {
            'w_angle': 1.0,
            'w_l1': 0.0,
            'w_l2': 0.0,
            'w_inf': 0.0,
            'w_mag': 0.0
        }

    def set_similarity_params(self, params: dict):
        """Sets parameters for the optimized similarity function."""
        self.similarity_params = params
        logger.info(f"Set similarity parameters: {params}")

    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generates an embedding for a text query.

        Args:
            text: The text query.

        Returns:
            Embedding as a numpy array (not normalized to preserve norm information).

        Raises:
            ValueError: If the text query is empty.
        """
        if not text.strip():
            raise ValueError("Text query cannot be empty")
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
        return text_features.cpu().numpy()[0]  # Return unnormalized embedding

    def search(self, text_query: str, top_k: int = 5, score_threshold: float = SCORE_THRESHOLD,
               use_optimized_similarity: bool = False):
        """Searches for images matching the text query with enhanced metrics.

        Args:
            text_query: The text to search for.
            top_k: Number of top results to return.
            score_threshold: Minimum similarity score for matches.
            use_optimized_similarity: Whether to rerank results using optimized similarity.

        Returns:
            List of dictionaries with image paths and scores.
        """
        logger.info(f"Searching for: {text_query} (optimized similarity: {use_optimized_similarity})")
        text_embedding = self.generate_text_embedding(text_query)

        # Normalize for Milvus search which uses cosine similarity
        text_embedding_normalized = text_embedding / np.linalg.norm(text_embedding)

        self.collection.load()
        try:
            # First pass: get candidates using standard cosine similarity
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[text_embedding_normalized],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 3,  # Get more candidates for reranking
                output_fields=["image_path", "embedding"]
            )

            matches = []
            for hits in results:
                for hit in hits:
                    path = hit.entity.get("image_path")
                    embedding = np.array(hit.entity.get("embedding"))

                    if use_optimized_similarity:
                        # Use optimized similarity for scoring
                        score = self.metrics.optimized_similarity(
                            text_embedding, embedding, self.similarity_params
                        )
                    else:
                        # Use standard cosine similarity
                        score = hit.score

                    matches.append({"path": path, "score": score, "embedding": embedding})

            # Sort by score and filter by threshold
            matches.sort(key=lambda x: x["score"], reverse=True)

            # Filter by threshold
            if use_optimized_similarity:
                # Adjust threshold for optimized similarity (may need different threshold)
                min_score = min(m["score"] for m in matches) if matches else 0
                max_score = max(m["score"] for m in matches) if matches else 1
                normalized_threshold = min_score + score_threshold * (max_score - min_score)
                filtered_matches = [m for m in matches if m["score"] >= normalized_threshold]
            else:
                filtered_matches = [m for m in matches if m["score"] >= score_threshold]

            # Ensure uniqueness
            seen_paths = set()
            unique_matches = []
            for match in filtered_matches:
                if match["path"] not in seen_paths:
                    seen_paths.add(match["path"])
                    # Remove embedding from result to keep output clean
                    del match["embedding"]
                    unique_matches.append(match)
                    if len(unique_matches) >= top_k:
                        break

            logger.info(f"Found {len(unique_matches)} matches for '{text_query}'")
            return unique_matches[:top_k]
        finally:
            self.collection.release()

    def search_with_multiple_metrics(self, text_query: str, top_k: int = 5):
        """
        Searches with multiple metrics and returns detailed comparison.

        Args:
            text_query: The text to search for.
            top_k: Number of top results to return.

        Returns:
            Dictionary with results for each metric.
        """
        logger.info(f"Multi-metric search for: {text_query}")
        text_embedding = self.generate_text_embedding(text_query)

        # Normalize for Milvus search
        text_embedding_normalized = text_embedding / np.linalg.norm(text_embedding)

        self.collection.load()
        try:
            # Get candidates using standard cosine similarity
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[text_embedding_normalized],
                anns_field="embedding",
                param=search_params,
                limit=top_k * 5,  # Get more candidates for analysis
                output_fields=["image_path", "embedding"]
            )

            candidates = []
            for hits in results:
                for hit in hits:
                    path = hit.entity.get("image_path")
                    embedding = np.array(hit.entity.get("embedding"))

                    # Calculate all distance metrics
                    distances = self.metrics.get_all_distances(text_embedding, embedding)

                    candidates.append({
                        "path": path,
                        "cosine_similarity": distances["cosine_similarity"],
                        "angular_distance": distances["angular_distance"],
                        "l1_distance": distances["l1_distance"],
                        "l2_distance": distances["l2_distance"],
                        "linf_distance": distances["linf_distance"],
                        "magnitude_difference": distances["magnitude_difference"],
                        "optimized_similarity": self.metrics.optimized_similarity(
                            text_embedding, embedding, self.similarity_params
                        )
                    })

            # Create results for each metric
            metric_results = {}

            # Cosine similarity results (standard)
            cosine_matches = sorted(candidates, key=lambda x: x["cosine_similarity"], reverse=True)[:top_k]
            metric_results["cosine_similarity"] = cosine_matches

            # L1 distance results
            l1_matches = sorted(candidates, key=lambda x: x["l1_distance"])[:top_k]
            metric_results["l1_distance"] = l1_matches

            # L2 distance results
            l2_matches = sorted(candidates, key=lambda x: x["l2_distance"])[:top_k]
            metric_results["l2_distance"] = l2_matches

            # L-infinity distance results
            linf_matches = sorted(candidates, key=lambda x: x["linf_distance"])[:top_k]
            metric_results["linf_distance"] = linf_matches

            # Magnitude difference results
            mag_matches = sorted(candidates, key=lambda x: x["magnitude_difference"])[:top_k]
            metric_results["magnitude_difference"] = mag_matches

            # Optimized similarity results
            optimized_matches = sorted(candidates, key=lambda x: x["optimized_similarity"], reverse=True)[:top_k]
            metric_results["optimized_similarity"] = optimized_matches

            # Calculate intersection metrics
            metric_results["analysis"] = self._analyze_metric_results(metric_results)

            logger.info(f"Completed multi-metric search for '{text_query}'")
            return metric_results
        finally:
            self.collection.release()

    def _analyze_metric_results(self, metric_results):
        """Analyzes the results from different metrics to provide insights."""
        analysis = {}

        # Get paths for each metric
        paths_by_metric = {}
        for metric, results in metric_results.items():
            if metric != "analysis":
                paths_by_metric[metric] = [r["path"] for r in results]

        # Calculate intersection between metrics
        intersections = {}
        for metric1 in paths_by_metric:
            for metric2 in paths_by_metric:
                if metric1 < metric2:  # Avoid duplicates
                    intersection = set(paths_by_metric[metric1]) & set(paths_by_metric[metric2])
                    intersections[f"{metric1}_vs_{metric2}"] = {
                        "intersection_size": len(intersection),
                        "intersection_ratio": len(intersection) / len(paths_by_metric[metric1]),
                        "common_items": list(intersection)
                    }

        analysis["intersections"] = intersections

        # Calculate unique contributions of each metric
        unique_contributions = {}
        for metric, paths in paths_by_metric.items():
            other_paths = set()
            for other_metric, other_metric_paths in paths_by_metric.items():
                if other_metric != metric:
                    other_paths.update(other_metric_paths)

            unique_paths = set(paths) - other_paths
            unique_contributions[metric] = {
                "unique_count": len(unique_paths),
                "unique_ratio": len(unique_paths) / len(paths) if paths else 0,
                "unique_items": list(unique_paths)
            }

        analysis["unique_contributions"] = unique_contributions

        return analysis

    def compare_search_methods(self, text_query: str, top_k: int = 5):
        """
        Performs search with both standard and optimized similarity for comparison.

        Args:
            text_query: The text to search for.
            top_k: Number of top results to return.

        Returns:
            Dictionary with results from both methods and comparison metrics.
        """
        # Standard search
        standard_results = self.search(text_query, top_k, use_optimized_similarity=False)

        # Optimized search
        optimized_results = self.search(text_query, top_k, use_optimized_similarity=True)

        # Compare results
        standard_paths = [r["path"] for r in standard_results]
        optimized_paths = [r["path"] for r in optimized_results]

        intersection = set(standard_paths) & set(optimized_paths)

        comparison = {
            "standard_results": standard_results,
            "optimized_results": optimized_results,
            "metrics": {
                "intersection_size": len(intersection),
                "intersection_ratio": len(intersection) / top_k if top_k > 0 else 0,
                "unique_to_standard": list(set(standard_paths) - set(optimized_paths)),
                "unique_to_optimized": list(set(optimized_paths) - set(standard_paths))
            }
        }

        logger.info(f"Comparison for '{text_query}': {len(intersection)}/{top_k} results in common")
        return comparison