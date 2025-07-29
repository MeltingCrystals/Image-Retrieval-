# geometric_metrics.py
"""Module for geometric similarity metrics beyond angular distance."""

import numpy as np
from typing import Dict, Tuple, Callable, List, Optional, Union


class GeometricSimilarityMetrics:
    """Class implementing various geometric similarity metrics for embeddings."""

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes the cosine similarity between two vectors."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    @staticmethod
    def angular_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes the angular distance (in radians) between two vectors."""
        cos_sim = GeometricSimilarityMetrics.cosine_similarity(vec1, vec2)
        # Clip to handle floating point imprecision
        cos_sim = np.clip(cos_sim, -1.0, 1.0)
        return np.arccos(cos_sim)

    @staticmethod
    def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes cosine distance (1 - cosine similarity)."""
        return 1.0 - GeometricSimilarityMetrics.cosine_similarity(vec1, vec2)

    @staticmethod
    def l1_distance(vec1: np.ndarray, vec2: np.ndarray, normalized: bool = True) -> float:
        """Computes the L1 (Manhattan) distance between two vectors."""
        distance = np.sum(np.abs(vec1 - vec2))
        if normalized:
            distance /= len(vec1)  # Normalize by dimension
        return distance

    @staticmethod
    def l2_distance(vec1: np.ndarray, vec2: np.ndarray, normalized: bool = True) -> float:
        """Computes the L2 (Euclidean) distance between two vectors."""
        distance = np.sqrt(np.sum((vec1 - vec2) ** 2))
        if normalized:
            distance /= np.sqrt(len(vec1))  # Normalize by sqrt of dimension
        return distance

    @staticmethod
    def linf_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes the L-infinity (Chebyshev) distance between two vectors."""
        return np.max(np.abs(vec1 - vec2))

    @staticmethod
    def magnitude_difference(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Computes the absolute difference in vector magnitudes."""
        return abs(np.linalg.norm(vec1) - np.linalg.norm(vec2))

    @staticmethod
    def optimized_similarity(vec1: np.ndarray, vec2: np.ndarray, params: Dict[str, float]) -> float:
        """
        Computes a weighted combination of various similarity metrics.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            params: Dictionary with weights for different components:
                   - w_angle: Weight for angular component (cosine similarity)
                   - w_l1: Weight for L1 distance
                   - w_l2: Weight for L2 distance
                   - w_inf: Weight for L-infinity distance
                   - w_mag: Weight for magnitude difference

        Returns:
            Combined similarity score (higher is more similar)
        """
        # Get default parameters if not provided
        w_angle = params.get('w_angle', 1.0)
        w_l1 = params.get('w_l1', 0.0)
        w_l2 = params.get('w_l2', 0.0)
        w_inf = params.get('w_inf', 0.0)
        w_mag = params.get('w_mag', 0.0)

        # Calculate separate components
        angle_component = w_angle * GeometricSimilarityMetrics.cosine_similarity(vec1, vec2)
        l1_component = w_l1 * GeometricSimilarityMetrics.l1_distance(vec1, vec2)
        l2_component = w_l2 * GeometricSimilarityMetrics.l2_distance(vec1, vec2)
        linf_component = w_inf * GeometricSimilarityMetrics.linf_distance(vec1, vec2)
        mag_component = w_mag * GeometricSimilarityMetrics.magnitude_difference(vec1, vec2)

        # Higher distance = lower similarity, so subtract these components
        similarity = angle_component - l1_component - l2_component - linf_component - mag_component

        return similarity

    @staticmethod
    def optimized_distance(vec1: np.ndarray, vec2: np.ndarray, params: Dict[str, float]) -> float:
        """
        Computes a weighted combination of various distance metrics.
        This is the distance version of optimized_similarity (lower is more similar).

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            params: Dictionary with weights for different components

        Returns:
            Combined distance score (lower is more similar)
        """
        # Simply negate the similarity to get a distance
        return -GeometricSimilarityMetrics.optimized_similarity(vec1, vec2, params)

    @staticmethod
    def get_all_metrics(vec1: np.ndarray, vec2: np.ndarray) -> Dict[str, float]:
        """
        Computes all distance/similarity metrics between two vectors.

        Returns:
            Dictionary with all distance/similarity metrics
        """
        return {
            'cosine_similarity': GeometricSimilarityMetrics.cosine_similarity(vec1, vec2),
            'cosine_distance': GeometricSimilarityMetrics.cosine_distance(vec1, vec2),
            'angular_distance': GeometricSimilarityMetrics.angular_distance(vec1, vec2),
            'l1_distance': GeometricSimilarityMetrics.l1_distance(vec1, vec2),
            'l2_distance': GeometricSimilarityMetrics.l2_distance(vec1, vec2),
            'linf_distance': GeometricSimilarityMetrics.linf_distance(vec1, vec2),
            'magnitude_difference': GeometricSimilarityMetrics.magnitude_difference(vec1, vec2)
        }

    @staticmethod
    def create_parameter_grid(granularity: int = 5) -> Dict[str, List[float]]:
        """
        Creates a grid of parameter values for optimization.

        Args:
            granularity: Number of values to try for each parameter

        Returns:
            Dictionary mapping parameter names to lists of values
        """
        values = np.linspace(0.0, 1.0, granularity)
        return {
            'w_angle': list(values),
            'w_l1': list(values),
            'w_l2': list(values),
            'w_inf': list(values),
            'w_mag': list(values)
        }