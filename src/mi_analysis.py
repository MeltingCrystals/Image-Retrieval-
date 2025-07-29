# Enhanced mi_analysis.py
"""Enhanced module for geometric information theory analysis of embeddings."""
import json
import logging
import os
from collections import defaultdict

import numpy as np
from pathlib import Path
import random

import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Tuple, Dict, Any, Optional
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from itertools import product

from tqdm import tqdm

from geometric_metrics import GeometricSimilarityMetrics

logger = logging.getLogger(__name__)


class MIAnalysis:
    """Handles mutual information analysis for image embeddings."""

    def __init__(self, embeddings: List[Tuple[str, np.ndarray]], num_pairs: int = 1000, num_bins: int = 20):
        """
        Initializes the MI analysis.

        Args:
            embeddings: List of (image_path, embedding) tuples.
            num_pairs: Number of image pairs to sample.
            num_bins: Number of bins for discretizing angles.
        """
        self.embeddings = embeddings
        self.num_pairs = min(num_pairs, len(embeddings) * (len(embeddings) - 1) // 2)
        self.num_bins = num_bins
        self.pairs = []
        self.angles = []
        self.labels = []
        self.label_map = {"same_object": 0, "same_category": 1, "different_categories": 2}
        self.label_names = ["Same Object", "Same Category", "Different Categories"]

    # In mi_analysis.py, add this method to your MIAnalysis class

    def generate_pairs(self):
        """Generate pairs based on embedding similarities (COCO-style)."""
        import random
        random.seed(42)

        if len(self.embeddings) < 10:
            logger.warning("Not enough embeddings for meaningful analysis")
            return

        logger.info(f"Generating pairs from {len(self.embeddings)} embeddings...")

        # Calculate all pairwise similarities (sample subset for large datasets)
        max_comparisons = min(50000, len(self.embeddings) * (len(self.embeddings) - 1) // 2)
        pairs_data = []

        # Sample pairs to avoid memory issues with large datasets
        indices = list(range(len(self.embeddings)))
        sampled_pairs = []

        for _ in range(max_comparisons):
            i, j = random.sample(indices, 2)
            if i > j:
                i, j = j, i  # Ensure i < j
            if (i, j) not in sampled_pairs:
                sampled_pairs.append((i, j))

        logger.info(f"Calculating similarities for {len(sampled_pairs)} pairs...")

        for i, j in sampled_pairs:
            path1, emb1 = self.embeddings[i]
            path2, emb2 = self.embeddings[j]

            # Calculate cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            pairs_data.append((i, j, similarity))

        # Sort by similarity
        pairs_data.sort(key=lambda x: x[2], reverse=True)

        # Divide into semantic categories based on similarity percentiles
        similarities = [p[2] for p in pairs_data]
        high_sim_threshold = np.percentile(similarities, 80)
        med_sim_threshold = np.percentile(similarities, 50)

        logger.info(f"Similarity thresholds: high={high_sim_threshold:.3f}, med={med_sim_threshold:.3f}")

        # Sample pairs from different similarity ranges
        num_pairs_per_type = min(self.num_pairs // 3, len(pairs_data) // 3)

        # High similarity pairs (same category)
        high_sim_pairs = [p for p in pairs_data if p[2] >= high_sim_threshold]
        sampled_high = random.sample(high_sim_pairs, min(num_pairs_per_type, len(high_sim_pairs)))

        # Medium similarity pairs (related category)
        med_sim_pairs = [p for p in pairs_data if med_sim_threshold <= p[2] < high_sim_threshold]
        sampled_med = random.sample(med_sim_pairs, min(num_pairs_per_type, len(med_sim_pairs)))

        # Low similarity pairs (different categories)
        low_sim_pairs = [p for p in pairs_data if p[2] < med_sim_threshold]
        sampled_low = random.sample(low_sim_pairs, min(num_pairs_per_type, len(low_sim_pairs)))

        # Create pairs and labels
        for i, j, sim in sampled_high:
            path1, emb1 = self.embeddings[i]
            path2, emb2 = self.embeddings[j]
            angle = self.compute_angle(emb1, emb2)

            self.pairs.append((path1, path2))
            self.angles.append(angle)
            self.labels.append("same_category")

        for i, j, sim in sampled_med:
            path1, emb1 = self.embeddings[i]
            path2, emb2 = self.embeddings[j]
            angle = self.compute_angle(emb1, emb2)

            self.pairs.append((path1, path2))
            self.angles.append(angle)
            self.labels.append("same_object")  # Use existing label

        for i, j, sim in sampled_low:
            path1, emb1 = self.embeddings[i]
            path2, emb2 = self.embeddings[j]
            angle = self.compute_angle(emb1, emb2)

            self.pairs.append((path1, path2))
            self.angles.append(angle)
            self.labels.append("different_categories")

        logger.info(f"Generated {len(self.pairs)} pairs for MI analysis: "
                    f"{self.labels.count('same_object')} same object, "
                    f"{self.labels.count('same_category')} same category, "
                    f"{self.labels.count('different_categories')} different categories.")

    def compute_angle(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes the angle between two embeddings in radians."""
        # Ensure vectors are normalized for accurate cosine similarity
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)

        cosine_sim = np.dot(emb1_norm, emb2_norm)
        cosine_sim = np.clip(cosine_sim, -1.0, 1.0)
        return np.arccos(cosine_sim)


class ColorMIAnalyzer:
    """
    Analyzer for mutual information between geometric properties of embeddings
    and color-based semantic relationships.
    """

    def __init__(self, base_dir: str = "color_dataset", bin_count: int = 20, bin_strategy: str = 'uniform'):
        """
        Initialize the analyzer.

        Args:
            base_dir: Directory containing the color dataset
            bin_count: Number of bins for discretizing continuous values
            bin_strategy: Strategy for binning ('uniform', 'quantile', etc.)
        """
        self.base_dir = Path(base_dir)
        self.bin_count = bin_count
        self.bin_strategy = bin_strategy
        self.metrics = GeometricSimilarityMetrics()

        self.relationship_types = [
            "same_object_same_color",  # Same object, same color (identical)
            "same_object_diff_color",  # Same object, different color
            "diff_object_same_color",  # Different objects, same color
            "diff_object_diff_color"  # Different objects, different color
        ]

        self.metric_names = [
            "cosine_distance",
            "l1_distance",
            "l2_distance",
            "linf_distance",
            "magnitude_difference"
        ]

        # Will store results of analyses
        self.embeddings = {}  # Image path -> embedding
        self.metadata = None  # DataFrame with image metadata
        self.pairs = {}  # Relationship type -> list of pairs
        self.distances = {}  # Metric -> relationship type -> distances
        self.mi_results = {}  # Various MI analysis results
        self.optimal_weights = {}  # Optimal weights for the similarity function

    def load_dataset(self, embeddings_file: str) -> Tuple[bool, str]:
        """
        Load the color dataset (metadata and pairs).

        Args:
            embeddings_file: Path to file containing embeddings

        Returns:
            Tuple of (success, message)
        """
        # Load metadata
        metadata_path = self.base_dir / "metadata.csv"
        if not metadata_path.exists():
            return False, f"Metadata file not found: {metadata_path}"

        self.metadata = pd.read_csv(metadata_path)
        logger.info(f"Loaded metadata for {len(self.metadata)} images")

        # Load pairs
        pairs_path = self.base_dir / "pairs.json"
        if not pairs_path.exists():
            return False, f"Pairs file not found: {pairs_path}"

        with open(pairs_path, 'r') as f:
            raw_pairs = json.load(f)

        # Convert to absolute paths
        for rel_type, rel_pairs in raw_pairs.items():
            self.pairs[rel_type] = []
            for p1, p2 in rel_pairs:
                path1 = os.path.join(self.base_dir, p1) if not os.path.isabs(p1) else p1
                path2 = os.path.join(self.base_dir, p2) if not os.path.isabs(p2) else p2
                self.pairs[rel_type].append((path1, path2))

        logger.info(
            f"Loaded pairs: {', '.join(f'{len(pairs)} {rel_type}' for rel_type, pairs in self.pairs.items())}")

        # Load embeddings
        try:
            # Attempt to load compressed npz file
            data = np.load(embeddings_file, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                # If it's a .npz file with multiple arrays
                if 'embeddings' in data:
                    self.embeddings = data['embeddings'].item()
                else:
                    return False, f"No 'embeddings' array found in {embeddings_file}"
            else:
                # If it's a .npy file with a single array
                self.embeddings = data.item()

            logger.info(f"Loaded embeddings for {len(self.embeddings)} images")
            return True, "Dataset loaded successfully"

        except Exception as e:
            return False, f"Error loading embeddings: {str(e)}"

    def calculate_distances(self) -> None:
        """
        Calculate distances between pairs using multiple metrics.
        """
        logger.info("Calculating distances for all pairs using multiple metrics")

        # Initialize distances dictionary
        self.distances = {}
        for metric_name in self.metric_names:
            self.distances[metric_name] = {}
            for rel_type in self.relationship_types:
                self.distances[metric_name][rel_type] = []

        # Calculate distances for each relationship type
        for rel_type in self.relationship_types:
            if rel_type not in self.pairs:
                logger.warning(f"No pairs found for relationship type: {rel_type}")
                continue

            logger.info(f"Processing {len(self.pairs[rel_type])} {rel_type} pairs")

            for img1_path, img2_path in tqdm(self.pairs[rel_type], desc=f"Computing {rel_type}"):
                # Get embeddings
                if img1_path not in self.embeddings or img2_path not in self.embeddings:
                    logger.warning(f"Embeddings not found for {img1_path} or {img2_path}")
                    continue

                v1 = self.embeddings[img1_path]
                v2 = self.embeddings[img2_path]

                # Calculate all metrics
                all_metrics = self.metrics.get_all_metrics(v1, v2)

                # Store each metric's distance
                for metric_name in self.metric_names:
                    if metric_name in all_metrics:
                        self.distances[metric_name][rel_type].append(all_metrics[metric_name])

        # Log the number of distances calculated
        for metric_name in self.metric_names:
            counts = {rel_type: len(distances) for rel_type, distances in self.distances[metric_name].items()}
            logger.info(f"{metric_name}: {counts}")

    def calculate_mutual_information(self) -> Dict[str, float]:
        """
        Calculate mutual information between distances and relationship types.

        Returns:
            Dictionary mapping metrics to their MI values
        """
        if not self.distances:
            self.calculate_distances()

        logger.info(f"Calculating mutual information with {self.bin_count} bins, {self.bin_strategy} strategy")

        mi_values = {}

        for metric_name in self.metric_names:
            # Collect all distances for this metric
            all_distances = []
            all_labels = []

            for i, rel_type in enumerate(self.relationship_types):
                # Add distances to the collection with their relationship label
                all_distances.extend(self.distances[metric_name][rel_type])
                all_labels.extend([i] * len(self.distances[metric_name][rel_type]))

            # Skip metrics with no valid distances
            if not all_distances:
                logger.warning(f"No valid distances found for {metric_name}, skipping MI calculation")
                mi_values[metric_name] = 0.0
                continue

            # Filter out NaN values
            valid_indices = [i for i, d in enumerate(all_distances) if not np.isnan(d)]
            valid_distances = [all_distances[i] for i in valid_indices]
            valid_labels = [all_labels[i] for i in valid_indices]

            if not valid_distances:
                logger.warning(f"No valid non-NaN distances found for {metric_name}, skipping MI calculation")
                mi_values[metric_name] = 0.0
                continue

            # Convert to numpy arrays
            X = np.array(valid_distances).reshape(-1, 1)
            y = np.array(valid_labels)

            # Bin the distances using the specified strategy
            discretizer = KBinsDiscretizer(n_bins=self.bin_count, encode='ordinal', strategy=self.bin_strategy)
            X_binned = discretizer.fit_transform(X).astype(int).flatten()

            # Calculate mutual information
            mi = mutual_info_score(X_binned, y)
            mi_values[metric_name] = mi

            logger.info(f"MI for {metric_name}: {mi:.4f} bits")

        self.mi_results['general'] = mi_values
        return mi_values

    def calculate_color_specific_mi(self) -> Dict[str, float]:
        """
        Calculate MI specifically for color discrimination within the same category.

        Returns:
            Dictionary mapping metrics to their color-specific MI values
        """
        logger.info("Calculating color-specific mutual information")

        # Get all pairs with same category, different colors
        color_mi = {}

        # We'll analyze "same_object_diff_color" pairs specifically
        if "same_object_diff_color" not in self.pairs:
            logger.warning("No 'same_object_diff_color' pairs found")
            return {}

        # The label is binary: same color (0) or different color (1)
        # For 'same_object_diff_color', all pairs should have different colors
        different_color_pairs = self.pairs["same_object_diff_color"]

        # Also include pairs from "same_object_same_color" for comparison
        if "same_object_same_color" in self.pairs:
            same_color_pairs = self.pairs["same_object_same_color"]
        else:
            same_color_pairs = []

        all_pairs = different_color_pairs + same_color_pairs
        all_labels = [1] * len(different_color_pairs) + [0] * len(same_color_pairs)

        # Calculate MI for each metric
        for metric_name in self.metric_names:
            # Calculate distances for all pairs
            distances = []
            valid_indices = []

            for i, (img1_path, img2_path) in enumerate(all_pairs):
                if img1_path not in self.embeddings or img2_path not in self.embeddings:
                    continue

                v1 = self.embeddings[img1_path]
                v2 = self.embeddings[img2_path]

                # Get all metrics
                all_metrics = self.metrics.get_all_metrics(v1, v2)

                if metric_name in all_metrics:
                    distances.append(all_metrics[metric_name])
                    valid_indices.append(i)

            # Get corresponding labels
            valid_labels = [all_labels[i] for i in valid_indices]

            # Skip metrics with no valid distances
            if not distances:
                logger.warning(f"No valid distances found for {metric_name} in color analysis, skipping MI calculation")
                color_mi[metric_name] = 0.0
                continue

            # Bin distances
            X = np.array(distances).reshape(-1, 1)
            y = np.array(valid_labels)

            discretizer = KBinsDiscretizer(n_bins=self.bin_count, encode='ordinal', strategy=self.bin_strategy)
            X_binned = discretizer.fit_transform(X).astype(int).flatten()

            # Calculate MI
            mi = mutual_info_score(X_binned, y)
            color_mi[metric_name] = mi

            logger.info(f"Color-specific MI for {metric_name}: {mi:.4f} bits")

        self.mi_results['color_specific'] = color_mi
        return color_mi

    def optimize_weights(self, grid_size: int = 5) -> Dict[str, float]:
        """
        Find optimal weights for combining metrics to maximize color discrimination.

        Args:
            grid_size: Number of values to try for each parameter

        Returns:
            Dictionary mapping metrics to optimal weights
        """
        logger.info(f"Optimizing weights with grid size {grid_size}")

        # Get all pairs with same category, different colors
        different_color_pairs = self.pairs["same_object_diff_color"]
        same_color_pairs = self.pairs["same_object_same_color"]

        all_pairs = different_color_pairs + same_color_pairs
        all_labels = [1] * len(different_color_pairs) + [0] * len(same_color_pairs)

        # Check if we have any valid pairs
        valid_pair_count = 0
        for img1_path, img2_path in all_pairs:
            if img1_path in self.embeddings and img2_path in self.embeddings:
                valid_pair_count += 1

        if valid_pair_count == 0:
            logger.warning("No valid pairs found for parameter optimization")
            # Return default weights
            return {
                'w_angle': 1.0,
                'w_l1': 0.0,
                'w_l2': 0.0,
                'w_inf': 0.0,
                'w_mag': 0.0
            }

        # Create parameter grid
        param_grid = {
            'w_angle': np.linspace(0.0, 1.0, grid_size),
            'w_l1': np.linspace(0.0, 1.0, grid_size),
            'w_l2': np.linspace(0.0, 1.0, grid_size),
            'w_inf': np.linspace(0.0, 1.0, grid_size),
            'w_mag': np.linspace(0.0, 1.0, grid_size)
        }

        # Create all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        # Evaluate each combination
        best_mi = -float('inf')
        best_params = {}

        logger.info(f"Starting grid search over {len(param_combinations)} parameter combinations...")

        for i, combination in enumerate(param_combinations):
            params = {name: value for name, value in zip(param_names, combination)}

            try:
                mi = self._evaluate_weights(all_pairs, all_labels, params)

                if mi > best_mi:
                    best_mi = mi
                    best_params = params.copy()

                if i % 100 == 0 or i == len(param_combinations) - 1:
                    logger.info(f"Evaluated {i + 1}/{len(param_combinations)} combinations. Best MI: {best_mi:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating parameters {params}: {e}")

        self.optimal_weights = best_params
        logger.info(f"Optimal parameters found: {best_params}, MI: {best_mi:.4f}")

        self.mi_results['optimized'] = {
            'parameters': best_params,
            'mi_value': best_mi
        }

        return best_params

    def _evaluate_weights(self, pairs: List[Tuple[str, str]],
                          labels: List[int],
                          weights: Dict[str, float]) -> float:
        """
        Evaluate weights by calculating MI between weighted distances and labels.

        Args:
            pairs: List of image pairs
            labels: List of labels (1 for different color, 0 for same color)
            weights: Dictionary mapping metrics to weights

        Returns:
            Mutual information value
        """
        # Calculate weighted distances
        distances = []
        valid_indices = []

        for i, (img1_path, img2_path) in enumerate(pairs):
            if img1_path not in self.embeddings or img2_path not in self.embeddings:
                continue

            v1 = self.embeddings[img1_path]
            v2 = self.embeddings[img2_path]

            # Calculate weighted distance
            distance = self.metrics.optimized_distance(v1, v2, weights)
            distances.append(distance)
            valid_indices.append(i)

        if not distances:
            return -float('inf')

        # Get corresponding labels
        valid_labels = [labels[i] for i in valid_indices]

        # Bin distances
        X = np.array(distances).reshape(-1, 1)
        y = np.array(valid_labels)

        discretizer = KBinsDiscretizer(n_bins=self.bin_count, encode='ordinal', strategy=self.bin_strategy)
        X_binned = discretizer.fit_transform(X).astype(int).flatten()

        # Calculate MI
        mi = mutual_info_score(X_binned, y)
        return mi

    def visualize_angle_distributions(self, output_path: Optional[str] = None) -> Figure:
        plt.figure(figsize=(12, 8))

        # Get angle data for each relationship type
        angle_data = {}
        for rel_type in self.relationship_types:
            if rel_type in self.distances.get('cosine_distance', {}):
                angles = []
                for d in self.distances['cosine_distance'][rel_type]:
                    arccos_input = 1 - min(d, 1.999)
                    if arccos_input > 1.0:
                        arccos_input = 1.0
                    elif arccos_input < -1.0:
                        arccos_input = -1.0
                    angles.append(np.arccos(arccos_input))
                angle_data[rel_type] = angles

        # Check if we have any data
        if not angle_data or all(len(angles) == 0 for angles in angle_data.values()):
            plt.text(0.5, 0.5, "No angle data available",
                     ha='center', va='center', transform=plt.gca().transAxes)
            logger.warning("No angle data available for visualization")
        else:
            # Print stats to debug
            for rel_type, angles in angle_data.items():
                if angles:
                    logger.info(f"{rel_type}: {len(angles)} angles, min={min(angles):.4f}, max={max(angles):.4f}")

            # Create histograms with auto-scaled bins and alpha for visibility
            for rel_type, angles in angle_data.items():
                if angles:
                    label = rel_type.replace('_', ' ').title()
                    plt.hist(angles, bins=20, alpha=0.7, label=label, density=True)

            # Set axis limits based on data
            all_angles = [angle for angles in angle_data.values() for angle in angles]
            if all_angles:
                min_angle, max_angle = min(all_angles), max(all_angles)
                # Add 10% padding
                padding = (max_angle - min_angle) * 0.1
                plt.xlim(max(0, min_angle - padding), max_angle + padding)

            # Calculate MI and add annotation after ensuring valid data
            if all_angles:
                # Calculate MI for angles
                all_angles_mi = []  # Renamed to avoid variable reuse
                all_labels = []
                for i, rel_type in enumerate(self.relationship_types):
                    if rel_type in angle_data:  # Fixed: check angle_data not angles
                        all_angles_mi.extend(angle_data[rel_type])  # Fixed: use angle_data
                        all_labels.extend([i] * len(angle_data[rel_type]))  # Fixed: use angle_data

                # Filter out any NaN values that might have slipped through
                valid_indices = ~np.isnan(np.array(all_angles_mi))  # Fixed: convert to numpy array
                valid_angles = np.array(all_angles_mi)[valid_indices]
                valid_labels = np.array(all_labels)[valid_indices]

                if len(valid_angles) == 0:
                    logger.warning("No valid angles for MI calculation")
                    plt.annotate("No valid MI calculation possible", xy=(0.5, 0.5),
                                 xycoords='axes fraction', ha='center', fontsize=12)
                else:
                    X = valid_angles.reshape(-1, 1)
                    y = valid_labels

                    discretizer = KBinsDiscretizer(n_bins=self.bin_count, encode='ordinal', strategy=self.bin_strategy)
                    X_binned = discretizer.fit_transform(X).astype(int).flatten()

                    mi = mutual_info_score(X_binned, y)
                    plt.annotate(f"MI: {mi:.4f} bits", xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12)

        plt.xlabel('Angle (radians)')
        plt.ylabel('Frequency')
        plt.title('Angle Distribution by Semantic Relationship')
        plt.legend()

        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Angle distribution visualization saved to {output_path}")

        fig = plt.gcf()  # Get current figure
        return fig

    def visualize_color_metrics_comparison(self, output_path: Optional[str] = None) -> Figure:
        """
        Visualize the comparison of color-specific MI across metrics.

        Args:
            output_path: Optional path to save the figure

        Returns:
            Figure object
        """
        if 'color_specific' not in self.mi_results:
            self.calculate_color_specific_mi()

        color_mi = self.mi_results['color_specific']

        plt.figure(figsize=(10, 6))

        metrics = list(color_mi.keys())
        mi_values = [color_mi[metric] for metric in metrics]

        # Sort by MI value
        sorted_indices = np.argsort(mi_values)[::-1]  # Descending order
        sorted_metrics = [metrics[i] for i in sorted_indices]
        sorted_values = [mi_values[i] for i in sorted_indices]

        bars = plt.bar(range(len(sorted_metrics)), sorted_values)

        # Color the bars
        color_map = plt.cm.viridis(np.linspace(0, 1, len(sorted_metrics)))
        for i, bar in enumerate(bars):
            bar.set_color(color_map[i])

        plt.xticks(range(len(sorted_metrics)), [m.replace('_', ' ').title() for m in sorted_metrics], rotation=45)
        plt.xlabel('Metric')
        plt.ylabel('Mutual Information (bits)')
        plt.title('Color Discrimination Power by Geometric Metric')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Metric comparison visualization saved to {output_path}")

        fig = plt.gcf()  # Get current figure
        return fig

    def visualize_metric_density_plots(self, output_dir: Optional[str] = None) -> List[Figure]:
        """
        Create density plots for each metric showing the distribution for different relationship types.

        Args:
            output_dir: Optional directory to save the figures

        Returns:
            List of Figure objects
        """
        if not self.distances:
            raise ValueError("No distances calculated. Run calculate_distances() first.")

        figures = []

        for metric_name in self.metric_names:
            plt.figure(figsize=(10, 6))

            # Create KDE plots for each relationship type
            for rel_type in self.relationship_types:
                if rel_type not in self.distances[metric_name]:
                    continue

                values = self.distances[metric_name][rel_type]
                sns.kdeplot(values, label=rel_type.replace('_', ' ').title())

            plt.xlabel(f'{metric_name.replace("_", " ").title()}')
            plt.ylabel('Density')
            plt.title(f'{metric_name.title()} Distribution by Relationship Type')
            plt.legend()

            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f'{metric_name}_density.png')
                plt.savefig(output_path, dpi=300)
                logger.info(f"{metric_name} density plot saved to {output_path}")

            figures.append(plt.gcf())  # Get current figure

        return figures

    def visualize_precision_recall_curves(self, output_path: Optional[str] = None) -> Figure:
        """
        Visualize precision-recall curves for color discrimination using different thresholds.

        Args:
            output_path: Optional path to save the figure

        Returns:
            Figure object
        """
        # Get pairs for color discrimination analysis
        different_color_pairs = self.pairs["same_object_diff_color"]
        same_color_pairs = self.pairs["same_object_same_color"]

        all_pairs = different_color_pairs + same_color_pairs
        all_labels = [1] * len(different_color_pairs) + [0] * len(same_color_pairs)

        # Calculate distances using both cosine and optimized metrics (if available)
        cosine_distances = []
        optimized_distances = []
        valid_indices = []

        for i, (img1_path, img2_path) in enumerate(all_pairs):
            if img1_path not in self.embeddings or img2_path not in self.embeddings:
                continue

            v1 = self.embeddings[img1_path]
            v2 = self.embeddings[img2_path]

            # Calculate cosine distance
            cosine_dist = self.metrics.cosine_distance(v1, v2)
            cosine_distances.append(cosine_dist)

            # Calculate optimized distance if weights are available
            if self.optimal_weights:
                opt_dist = self.metrics.optimized_distance(v1, v2, self.optimal_weights)
                optimized_distances.append(opt_dist)

            valid_indices.append(i)

        # Get corresponding labels
        valid_labels = [all_labels[i] for i in valid_indices]

        # Calculate precision-recall for different thresholds
        thresholds = np.linspace(0, 1, 100)

        plt.figure(figsize=(10, 6))

        # Function to calculate precision and recall for a set of distances
        def calculate_pr_curve(distances, name):
            precision_values = []
            recall_values = []

            for threshold in thresholds:
                predictions = [1 if d <= threshold else 0 for d in distances]

                # Calculate confusion matrix values
                tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, valid_labels))
                fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, valid_labels))
                fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, valid_labels))

                # Calculate precision and recall
                precision = tp / (tp + fp) if tp + fp > 0 else 0
                recall = tp / (tp + fn) if tp + fn > 0 else 0

                precision_values.append(precision)
                recall_values.append(recall)

            return precision_values, recall_values

        # Calculate and plot for cosine distance
        cosine_precision, cosine_recall = calculate_pr_curve(cosine_distances, "Cosine")
        plt.plot(thresholds, cosine_precision, 'b-', label='Cosine Precision')
        plt.plot(thresholds, cosine_recall, 'b--', label='Cosine Recall')

        # Calculate F1 score to find optimal threshold
        cosine_f1 = [2 * p * r / (p + r) if p + r > 0 else 0
                     for p, r in zip(cosine_precision, cosine_recall)]
        cosine_best_idx = np.argmax(cosine_f1)
        cosine_best_threshold = thresholds[cosine_best_idx]

        # Add vertical line for optimal threshold
        plt.axvline(cosine_best_threshold, color='b', linestyle=':',
                    label=f'Cosine Optimal ({cosine_best_threshold:.2f})')

        # If we have optimized weights, plot those too
        if self.optimal_weights and optimized_distances:
            opt_precision, opt_recall = calculate_pr_curve(optimized_distances, "Optimized")
            plt.plot(thresholds, opt_precision, 'r-', label='Optimized Precision')
            plt.plot(thresholds, opt_recall, 'r--', label='Optimized Recall')

            # Calculate F1 score to find optimal threshold
            opt_f1 = [2 * p * r / (p + r) if p + r > 0 else 0
                      for p, r in zip(opt_precision, opt_recall)]
            opt_best_idx = np.argmax(opt_f1)
            opt_best_threshold = thresholds[opt_best_idx]

            # Add vertical line for optimal threshold
            plt.axvline(opt_best_threshold, color='r', linestyle=':',
                        label=f'Optimized Optimal ({opt_best_threshold:.2f})')

        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Precision-Recall vs. Threshold for Color Discrimination')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Precision-recall curve saved to {output_path}")

        fig = plt.gcf()
        return fig

    def visualize_bin_sensitivity(self, output_path: Optional[str] = None) -> Figure:
        """
        Visualize how MI varies with bin count for different metrics.

        Args:
            output_path: Optional path to save the figure

        Returns:
            Figure object
        """
        # Try different bin counts
        bin_counts = list(range(10, 51, 5))  # 10, 15, ..., 50

        # Store MI values for each metric and bin count
        mi_values = defaultdict(list)

        for bin_count in tqdm(bin_counts, desc="Testing bin sensitivity"):
            # Update bin count
            self.bin_count = bin_count

            # Calculate MI for different relationship types
            mi = self.calculate_mutual_information()

            # Store results
            for metric, value in mi.items():
                mi_values[metric].append(value)

        # Create plot
        plt.figure(figsize=(10, 6))

        for metric, values in mi_values.items():
            plt.plot(bin_counts, values, marker='o', label=metric.replace('_', ' ').title())

        plt.xlabel('Number of Bins')
        plt.ylabel('Mutual Information (bits)')
        plt.title('Bin Count Sensitivity Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Highlight optimal bin count
        for metric, values in mi_values.items():
            best_idx = np.argmax(values)
            best_bin_count = bin_counts[best_idx]
            best_mi = values[best_idx]

            plt.scatter([best_bin_count], [best_mi], color='red', s=100, zorder=5)
            plt.annotate(f"Optimal: {best_bin_count}",
                         xy=(best_bin_count, best_mi),
                         xytext=(best_bin_count + 2, best_mi + 0.01),
                         arrowprops=dict(arrowstyle="->", color="red"))

        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Bin sensitivity visualization saved to {output_path}")

        # Reset bin count to original value or optimal
        self.bin_count = 20  # Reset to default

        fig = plt.gcf()
        return fig

    def create_summary_visualization(self, output_path: Optional[str] = None) -> Figure:
        """
        Create a summary visualization of the key findings.

        Args:
            output_path: Optional path to save the figure

        Returns:
            Figure object
        """
        # Ensure we have all the necessary results
        if not self.mi_results.get('general'):
            self.calculate_mutual_information()

        if not self.mi_results.get('color_specific'):
            self.calculate_color_specific_mi()

        # Create figure with 2x2 grid of plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Angle distribution (top left)
        ax1 = axes[0, 0]

        # Convert cosine distances to angles
        angles = {}
        for rel_type in self.relationship_types:
            if rel_type not in self.distances.get('cosine_distance', {}):
                continue

            # Convert cosine distance (1-cos) to angle in radians with proper bounds checking
            angles[rel_type] = []
            for d in self.distances['cosine_distance'][rel_type]:
                # Ensure the value is within valid arccos range (-1 to 1)
                arccos_input = 1 - min(d, 1.999)
                if arccos_input > 1.0:
                    arccos_input = 1.0
                elif arccos_input < -1.0:
                    arccos_input = -1.0

                angles[rel_type].append(np.arccos(arccos_input))

        # Create histogram
        for rel_type in self.relationship_types:
            if rel_type not in angles:
                continue

            # Filter out NaN values
            valid_angles = [a for a in angles[rel_type] if not np.isnan(a)]
            if valid_angles:
                sns.histplot(valid_angles, bins=20, alpha=0.7,
                             label=rel_type.replace('_', ' ').title(), ax=ax1, kde=True)

        ax1.set_xlabel('Angle (radians)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Angle Distribution by Semantic Relationship')
        ax1.legend()

        # Calculate and display MI
        all_angles = []
        all_labels = []
        for i, rel_type in enumerate(self.relationship_types):
            if rel_type not in angles:
                continue

            # Filter out NaN values
            valid_angles = [a for a in angles[rel_type] if not np.isnan(a)]
            all_angles.extend(valid_angles)
            all_labels.extend([i] * len(valid_angles))

        if all_angles:
            X = np.array(all_angles).reshape(-1, 1)
            y = np.array(all_labels)

            discretizer = KBinsDiscretizer(n_bins=self.bin_count, encode='ordinal', strategy=self.bin_strategy)
            X_binned = discretizer.fit_transform(X).astype(int).flatten()

            mi = mutual_info_score(X_binned, y)
            ax1.annotate(f"MI: {mi:.4f} bits", xy=(0.7, 0.9), xycoords='axes fraction', fontsize=12)
        else:
            ax1.annotate("No valid angles for MI calculation", xy=(0.5, 0.5),
                         xycoords='axes fraction', ha='center', fontsize=12)

        # Plot 2: Metric comparison for color discrimination (top right)
        ax2 = axes[0, 1]

        color_mi = self.mi_results['color_specific']
        metrics = list(color_mi.keys())
        mi_values = [color_mi[metric] for metric in metrics]

        # Sort by MI value
        sorted_indices = np.argsort(mi_values)[::-1]  # Descending order
        sorted_metrics = [metrics[i] for i in sorted_indices]
        sorted_values = [mi_values[i] for i in sorted_indices]

        bars = ax2.bar(range(len(sorted_metrics)), sorted_values)

        # Color the bars
        color_map = plt.cm.viridis(np.linspace(0, 1, len(sorted_metrics)))
        for i, bar in enumerate(bars):
            bar.set_color(color_map[i])

        ax2.set_xticks(range(len(sorted_metrics)))
        ax2.set_xticklabels([m.replace('_', ' ').title() for m in sorted_metrics], rotation=45)
        ax2.set_xlabel('Metric')
        ax2.set_ylabel('Mutual Information (bits)')
        ax2.set_title('Color Discrimination Power by Geometric Metric')

        # Plot 3: Density plots for key metrics (bottom left)
        ax3 = axes[1, 0]

        # Choose top 2 metrics for clarity
        top_metrics = sorted_metrics[:2]

        for metric_name in top_metrics:
            for rel_type in ["same_object_same_color", "same_object_diff_color"]:
                if rel_type not in self.distances.get(metric_name, {}):
                    continue

                values = self.distances[metric_name][rel_type]
                label = f"{metric_name.replace('_', ' ').title()} - {rel_type.replace('_', ' ').title()}"
                sns.kdeplot(values, label=label, ax=ax3)

        ax3.set_xlabel('Distance Value')
        ax3.set_ylabel('Density')
        ax3.set_title('Distance Distribution for Color Comparison')
        ax3.legend()

        # Plot 4: Precision-Recall curve (bottom right)
        ax4 = axes[1, 1]

        # Get pairs for color discrimination analysis
        different_color_pairs = self.pairs.get("same_object_diff_color", [])
        same_color_pairs = self.pairs.get("same_object_same_color", [])

        if different_color_pairs and same_color_pairs:
            all_pairs = different_color_pairs + same_color_pairs
            all_labels = [1] * len(different_color_pairs) + [0] * len(same_color_pairs)

            # Calculate distances using both cosine and top metric
            cosine_distances = []
            top_metric_distances = []
            valid_indices = []

            for i, (img1_path, img2_path) in enumerate(all_pairs):
                if img1_path not in self.embeddings or img2_path not in self.embeddings:
                    continue

                v1 = self.embeddings[img1_path]
                v2 = self.embeddings[img2_path]

                # Calculate distances
                all_metrics = self.metrics.get_all_metrics(v1, v2)

                cosine_distances.append(all_metrics['cosine_distance'])

                if top_metrics[0] in all_metrics:
                    top_metric_distances.append(all_metrics[top_metrics[0]])

                valid_indices.append(i)

            # Get corresponding labels
            valid_labels = [all_labels[i] for i in valid_indices]

            # Calculate precision-recall for different thresholds
            thresholds = np.linspace(0, 1, 100)

            # Function to calculate precision and recall for a set of distances
            # In the precision-recall curve section, add filtering:

            # Function to calculate precision and recall for a set of distances
            def calculate_pr_curve(distances, name):
                precision_values = []
                recall_values = []

                for threshold in thresholds:
                    # Filter out NaN values
                    valid_indices = [i for i, d in enumerate(distances) if not np.isnan(d)]
                    valid_distances = [distances[i] for i in valid_indices]
                    valid_label_subset = [valid_labels[i] for i in valid_indices]

                    predictions = [1 if d <= threshold else 0 for d in valid_distances]

                    # Calculate confusion matrix values
                    tp = sum(pred == 1 and label == 1 for pred, label in zip(predictions, valid_label_subset))
                    fp = sum(pred == 1 and label == 0 for pred, label in zip(predictions, valid_label_subset))
                    fn = sum(pred == 0 and label == 1 for pred, label in zip(predictions, valid_label_subset))

                    # Calculate precision and recall
                    precision = tp / (tp + fp) if tp + fp > 0 else 0
                    recall = tp / (tp + fn) if tp + fn > 0 else 0

                    precision_values.append(precision)
                    recall_values.append(recall)

                return precision_values, recall_values

            # Calculate and plot for cosine distance
            cosine_precision, cosine_recall = calculate_pr_curve(cosine_distances, "Cosine")
            ax4.plot(thresholds, cosine_precision, 'b-', label='Cosine Precision')
            ax4.plot(thresholds, cosine_recall, 'b--', label='Cosine Recall')

            # Calculate F1 score to find optimal threshold
            cosine_f1 = [2 * p * r / (p + r) if p + r > 0 else 0
                         for p, r in zip(cosine_precision, cosine_recall)]
            cosine_best_idx = np.argmax(cosine_f1)
            cosine_best_threshold = thresholds[cosine_best_idx]

            # Add vertical line for optimal threshold
            ax4.axvline(cosine_best_threshold, color='b', linestyle=':',
                        label=f'Cosine Optimal ({cosine_best_threshold:.2f})')

            # If we have top metric distances, plot those too
            if top_metric_distances:
                top_precision, top_recall = calculate_pr_curve(top_metric_distances, top_metrics[0])
                ax4.plot(thresholds, top_precision, 'r-', label=f'{top_metrics[0].title()} Precision')
                ax4.plot(thresholds, top_recall, 'r--', label=f'{top_metrics[0].title()} Recall')

                # Calculate F1 score to find optimal threshold
                top_f1 = [2 * p * r / (p + r) if p + r > 0 else 0
                          for p, r in zip(top_precision, top_recall)]
                top_best_idx = np.argmax(top_f1)
                top_best_threshold = thresholds[top_best_idx]

                # Add vertical line for optimal threshold
                ax4.axvline(top_best_threshold, color='r', linestyle=':',
                            label=f'{top_metrics[0].title()} Optimal ({top_best_threshold:.2f})')

            ax4.set_xlabel('Threshold')
            ax4.set_ylabel('Score')
            ax4.set_title('Precision-Recall vs. Threshold for Color Discrimination')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, "Insufficient data for precision-recall analysis",
                     ha='center', va='center', fontsize=12)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            logger.info(f"Summary visualization saved to {output_path}")

        return fig

    def run_full_analysis(self, embeddings_file: str, output_dir: str = "results") -> Dict[str, Any]:
        """
        Run the complete analysis pipeline and save all results.

        Args:
            embeddings_file: Path to file containing embeddings
            output_dir: Directory to save results

        Returns:
            Dictionary with all results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Starting full analysis with embeddings from {embeddings_file}")

        # Step 1: Load dataset
        success, message = self.load_dataset(embeddings_file)
        if not success:
            logger.error(f"Failed to load dataset: {message}")
            return {"error": message}

        # Check if we have any valid embeddings
        if len(self.embeddings) == 0:
            error_msg = "No valid embeddings found in the embedding file"
            logger.error(error_msg)
            return {"error": error_msg}

        # Log some stats about the embeddings
        logger.info(f"Loaded {len(self.embeddings)} embeddings for analysis")

        # Step 2: Calculate distances
        self.calculate_distances()

        # Check if we have any valid distances
        has_valid_distances = False
        for metric_name in self.metric_names:
            for rel_type in self.relationship_types:
                if len(self.distances[metric_name][rel_type]) > 0:
                    has_valid_distances = True
                    break
            if has_valid_distances:
                break

        if not has_valid_distances:
            error_msg = "No valid distances could be calculated - path mismatch between embeddings and pairs"
            logger.error(error_msg)
            return {"error": error_msg}

        # Step 3: Calculate general mutual information
        general_mi = self.calculate_mutual_information()

        # Step 4: Calculate color-specific mutual information
        color_mi = self.calculate_color_specific_mi()

        # Step 5: Optimize weights
        optimal_weights = self.optimize_weights(grid_size=3)  # Use smaller grid for faster processing

        # Step 6: Create visualizations
        visualizations = {}

        # Angle distribution
        angle_fig = self.visualize_angle_distributions(
            os.path.join(output_dir, "angle_distribution.png"))
        visualizations["angle_distribution"] = angle_fig

        # Color metrics comparison
        color_metrics_fig = self.visualize_color_metrics_comparison(
            os.path.join(output_dir, "color_metrics_comparison.png"))
        visualizations["color_metrics_comparison"] = color_metrics_fig

        # Metric density plots
        density_figs = self.visualize_metric_density_plots(output_dir)
        visualizations["density_plots"] = density_figs

        # Precision-recall curves
        pr_fig = self.visualize_precision_recall_curves(
            os.path.join(output_dir, "precision_recall_curves.png"))
        visualizations["precision_recall_curves"] = pr_fig

        # Bin sensitivity - skip this as it's time-consuming
        # bin_fig = self.visualize_bin_sensitivity(
        #    os.path.join(output_dir, "bin_sensitivity.png"))
        # visualizations["bin_sensitivity"] = bin_fig

        # Summary visualization
        summary_fig = self.create_summary_visualization(
            os.path.join(output_dir, "summary.png"))
        visualizations["summary"] = summary_fig

        # Step 7: Save numerical results
        results = {
            "general_mi": general_mi,
            "color_mi": color_mi,
            "optimal_weights": optimal_weights
        }

        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            # Convert any numpy values to float for JSON serialization
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(i) for i in obj]
                else:
                    return obj

            json.dump(convert_to_serializable(results), f, indent=2)

        logger.info(f"Analysis complete. Results saved to {output_dir}")

        return {
            "results": results,
            "visualizations": visualizations
        }


class EnhancedMIAnalysis(MIAnalysis):
    """Enhanced MI analysis incorporating multiple geometric metrics."""

    def __init__(self, embeddings: List[Tuple[str, np.ndarray]], num_pairs: int = 1000, num_bins: int = 20,
                 keep_unnormalized: bool = True):
        """
        Initializes the enhanced MI analysis.

        Args:
            embeddings: List of (image_path, embedding) tuples.
            num_pairs: Number of image pairs to sample.
            num_bins: Number of bins for discretizing metrics.
            keep_unnormalized: If True, keeps original unnormalized embeddings for norm analysis.
        """
        super().__init__(embeddings, num_pairs, num_bins)
        self.keep_unnormalized = keep_unnormalized
        self.original_embeddings = []
        self.metrics = GeometricSimilarityMetrics()  # Initialize metrics

        # Store original unnormalized embeddings if requested
        if keep_unnormalized:
            self.original_embeddings = [(path, emb.copy()) for path, emb in embeddings]

        # Dictionary to store different distance measures
        self.distance_measures = {
            'angular': [],
            'l1': [],
            'l2': [],
            'linf': [],
            'magnitude_diff': []
        }

        # Store MI values for each metric
        self.mi_values = {}

        # Current optimal parameters for the similarity function
        self.optimal_params = {
            'w_angle': 1.0,
            'w_l1': 0.0,
            'w_l2': 0.0,
            'w_inf': 0.0,
            'w_mag': 0.0
        }

    def find_optimal_parameters(self, param_grid=None):
        """
        Find optimal parameters for the similarity function through grid search.

        Args:
            param_grid: Optional parameter grid, if None, a default grid is created

        Returns:
            Dictionary with parameters and MI value
        """
        # Create default parameter grid if none provided
        if param_grid is None:
            param_grid = {
                'w_angle': np.linspace(0, 1, 5),
                'w_l1': np.linspace(0, 1, 5),
                'w_l2': np.linspace(0, 1, 5),
                'w_inf': np.linspace(0, 1, 5),
                'w_mag': np.linspace(0, 1, 5)
            }

        # Create all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        # Evaluate each combination
        best_mi = -float('inf')
        best_params = {}

        logger.info(f"Starting grid search over {len(param_combinations)} parameter combinations...")

        for i, combination in enumerate(param_combinations):
            params = {name: value for name, value in zip(param_names, combination)}

            try:
                mi = self.compute_mi_for_optimized_similarity(params)

                if mi > best_mi:
                    best_mi = mi
                    best_params = params.copy()

                if i % 10 == 0 or i == len(param_combinations) - 1:
                    logger.info(f"Evaluated {i + 1}/{len(param_combinations)} combinations. Best MI: {best_mi:.4f}")

            except Exception as e:
                logger.error(f"Error evaluating parameters {params}: {e}")

        self.optimal_params = best_params
        logger.info(f"Optimal parameters found: {best_params}, MI: {best_mi:.4f}")

        return {
            'parameters': best_params,
            'mi_value': best_mi
        }

    def generate_coco_pairs(self):
        """Generate pairs based on COCO image similarities."""
        import random
        random.seed(42)

        if len(self.embeddings) < 10:
            logger.warning("Not enough embeddings for meaningful analysis")
            return

        # Calculate all pairwise similarities
        similarities = []
        pairs_data = []

        for i in range(len(self.embeddings)):
            for j in range(i + 1, len(self.embeddings)):
                path1, emb1 = self.embeddings[i]
                path2, emb2 = self.embeddings[j]

                # Calculate cosine similarity
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarities.append(similarity)
                pairs_data.append((i, j, similarity))

        # Sort by similarity
        pairs_data.sort(key=lambda x: x[2], reverse=True)

        # Divide into semantic categories based on similarity
        total_pairs = len(pairs_data)
        high_sim_threshold = np.percentile([p[2] for p in pairs_data], 80)
        med_sim_threshold = np.percentile([p[2] for p in pairs_data], 50)

        # Sample pairs from different similarity ranges
        num_pairs_per_type = min(self.num_pairs // 3, total_pairs // 3)

        # High similarity pairs (same category)
        high_sim_pairs = [p for p in pairs_data if p[2] >= high_sim_threshold]
        sampled_high = random.sample(high_sim_pairs, min(num_pairs_per_type, len(high_sim_pairs)))

        # Medium similarity pairs (related category)
        med_sim_pairs = [p for p in pairs_data if med_sim_threshold <= p[2] < high_sim_threshold]
        sampled_med = random.sample(med_sim_pairs, min(num_pairs_per_type, len(med_sim_pairs)))

        # Low similarity pairs (different categories)
        low_sim_pairs = [p for p in pairs_data if p[2] < med_sim_threshold]
        sampled_low = random.sample(low_sim_pairs, min(num_pairs_per_type, len(low_sim_pairs)))

        # Create pairs and labels
        for i, j, sim in sampled_high:
            path1, emb1 = self.embeddings[i]
            path2, emb2 = self.embeddings[j]
            angle = self.compute_angle(emb1, emb2)

            self.pairs.append((path1, path2))
            self.angles.append(angle)
            self.labels.append("same_category")

        for i, j, sim in sampled_med:
            path1, emb1 = self.embeddings[i]
            path2, emb2 = self.embeddings[j]
            angle = self.compute_angle(emb1, emb2)

            self.pairs.append((path1, path2))
            self.angles.append(angle)
            self.labels.append("related_category")

        for i, j, sim in sampled_low:
            path1, emb1 = self.embeddings[i]
            path2, emb2 = self.embeddings[j]
            angle = self.compute_angle(emb1, emb2)

            self.pairs.append((path1, path2))
            self.angles.append(angle)
            self.labels.append("different_categories")

        logger.info(f"Generated {len(self.pairs)} pairs for MI analysis: "
                    f"{self.labels.count('same_category')} same category, "
                    f"{self.labels.count('related_category')} related category, "
                    f"{self.labels.count('different_categories')} different categories.")

    def compute_mi_for_optimized_similarity(self, params: Dict[str, float]) -> float:
        """
        Compute mutual information using optimized similarity with the given parameters.

        Args:
            params: Parameters for the optimized similarity function (weights for different metrics)

        Returns:
            Mutual information value in bits
        """
        if not self.pairs:
            raise ValueError("No pairs generated. Run generate_pairs() first.")

        # Calculate optimized similarity scores for all pairs
        optimized_scores = []

        for i, (path1, path2) in enumerate(self.pairs):
            # Find the embeddings for the paths
            emb1 = None
            emb2 = None

            for p, emb in self.embeddings:
                if p == path1:
                    emb1 = emb
                elif p == path2:
                    emb2 = emb

            # Skip if we couldn't find the embeddings
            if emb1 is None or emb2 is None:
                continue

            # Calculate optimized similarity
            if self.keep_unnormalized:
                # Find original embeddings
                orig_emb1 = None
                orig_emb2 = None
                for p, emb in self.original_embeddings:
                    if p == path1:
                        orig_emb1 = emb
                    elif p == path2:
                        orig_emb2 = emb

                if orig_emb1 is not None and orig_emb2 is not None:
                    similarity = self.metrics.optimized_similarity(orig_emb1, orig_emb2, params)
                    optimized_scores.append(similarity)
            else:
                similarity = self.metrics.optimized_similarity(emb1, emb2, params)
                optimized_scores.append(similarity)

        # Convert labels to numeric
        numeric_labels = [self.label_map.get(label, -1) for label in self.labels[:len(optimized_scores)]]

        # Bin the optimized scores
        X = np.array(optimized_scores).reshape(-1, 1)
        y = np.array(numeric_labels)

        discretizer = KBinsDiscretizer(n_bins=self.num_bins, encode='ordinal', strategy='uniform')
        X_binned = discretizer.fit_transform(X).astype(int).flatten()

        # Calculate mutual information
        mi = mutual_info_score(X_binned, y)

        return mi

    def compute_mi_for_all_metrics(self) -> Dict[str, float]:
        """
        Compute mutual information between different geometric metrics and relationship labels.

        First calculates distances using various metrics, then computes mutual information
        between these distances and the semantic relationship labels.

        Returns:
            Dictionary mapping metrics to MI values
        """
        if not self.pairs:
            raise ValueError("No pairs generated. Run generate_pairs() first.")

        # Calculate distances using multiple metrics
        self.distance_measures = {
            'angular': [],
            'l1': [],
            'l2': [],
            'linf': [],
            'magnitude_diff': []
        }

        # Process each pair
        for i, (path1, path2) in enumerate(self.pairs):
            # Find the embeddings for the paths
            emb1 = None
            emb2 = None

            for p, emb in self.embeddings:
                if p == path1:
                    emb1 = emb
                elif p == path2:
                    emb2 = emb

            # Skip if we couldn't find the embeddings
            if emb1 is None or emb2 is None:
                continue

            # Use unnormalized embeddings if available
            if self.keep_unnormalized:
                orig_emb1 = None
                orig_emb2 = None
                for p, emb in self.original_embeddings:
                    if p == path1:
                        orig_emb1 = emb
                    elif p == path2:
                        orig_emb2 = emb

                if orig_emb1 is not None and orig_emb2 is not None:
                    # Calculate angular distance
                    angle = self.compute_angle(emb1, emb2)  # Using normalized embeddings for angle
                    self.distance_measures['angular'].append(angle)

                    # Calculate L1 distance
                    l1_dist = self.metrics.l1_distance(orig_emb1, orig_emb2)
                    self.distance_measures['l1'].append(l1_dist)

                    # Calculate L2 distance
                    l2_dist = self.metrics.l2_distance(orig_emb1, orig_emb2)
                    self.distance_measures['l2'].append(l2_dist)

                    # Calculate L-infinity distance
                    linf_dist = self.metrics.linf_distance(orig_emb1, orig_emb2)
                    self.distance_measures['linf'].append(linf_dist)

                    # Calculate magnitude difference
                    mag_diff = self.metrics.magnitude_difference(orig_emb1, orig_emb2)
                    self.distance_measures['magnitude_diff'].append(mag_diff)
            else:
                # Using only normalized embeddings
                # Calculate angular distance
                angle = self.compute_angle(emb1, emb2)
                self.distance_measures['angular'].append(angle)

                # Calculate L1 distance (with normalized embeddings)
                l1_dist = self.metrics.l1_distance(emb1, emb2)
                self.distance_measures['l1'].append(l1_dist)

                # Calculate L2 distance (with normalized embeddings)
                l2_dist = self.metrics.l2_distance(emb1, emb2)
                self.distance_measures['l2'].append(l2_dist)

                # Calculate L-infinity distance (with normalized embeddings)
                linf_dist = self.metrics.linf_distance(emb1, emb2)
                self.distance_measures['linf'].append(linf_dist)

                # Calculate magnitude difference (should be near 0 for normalized)
                mag_diff = self.metrics.magnitude_difference(emb1, emb2)
                self.distance_measures['magnitude_diff'].append(mag_diff)

        # Convert labels to numeric
        numeric_labels = [self.label_map.get(label, -1) for label in
                          self.labels[:len(self.distance_measures['angular'])]]

        # Calculate MI for each metric
        self.mi_values = {}

        for metric_name, distances in self.distance_measures.items():
            X = np.array(distances).reshape(-1, 1)
            y = np.array(numeric_labels)

            discretizer = KBinsDiscretizer(n_bins=self.num_bins, encode='ordinal', strategy='uniform')
            X_binned = discretizer.fit_transform(X).astype(int).flatten()

            mi = mutual_info_score(X_binned, y)
            self.mi_values[metric_name] = mi

            logger.info(f"MI for {metric_name}: {mi:.4f} bits")

        return self.mi_values



def analyze_color_embeddings(embeddings_file: str,
                             dataset_dir: str = "color_dataset",
                             output_dir: str = "results",
                             bin_count: int = 20) -> Dict[str, Any]:
    """
    Analyze embeddings using the color-based geometric information theory approach.

    Args:
        embeddings_file: Path to file containing embeddings
        dataset_dir: Directory containing the color dataset
        output_dir: Directory to save results
        bin_count: Number of bins for discretizing continuous values

    Returns:
        Dictionary with analysis results
    """
    analyzer = ColorMIAnalyzer(base_dir=dataset_dir, bin_count=bin_count)
    return analyzer.run_full_analysis(embeddings_file, output_dir)


