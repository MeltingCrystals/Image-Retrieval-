# app_pipeline.py
"""Pipeline for enhanced image search application."""

import logging
import numpy as np
import os
from pathlib import Path
from mi_analysis import EnhancedMIAnalysis
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EnhancedImageSearchApp:
    """Enhanced image search application with geometric metrics."""

    def __init__(self):
        self.embeddings = {}
        self.searcher = SimpleSearcher()

    def scan_folders(self):
        """Scan for image folders."""
        from tkinter import filedialog
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            return list(Path(folder).glob("*.jpg")) + list(Path(folder).glob("*.png"))
        return []

    def process_images(self, image_paths):
        """Process images and load real CLIP embeddings."""
        logger.info(f"Processing {len(image_paths)} images...")

        # Try to load existing embeddings first
        possible_embedding_paths = [
            "color_embeddings.npz",
            "color_analysis/color_embeddings.npz",
            "../color_embeddings.npz",
            "embeddings.npz",
            "color_dataset/embeddings.npz",
            os.path.expanduser("~/Desktop/color_embeddings.npz"),
            os.path.expanduser("~/Desktop/color_analysis/color_embeddings.npz")
        ]

        embeddings_file = None
        for path in possible_embedding_paths:
            if os.path.exists(path):
                embeddings_file = path
                logger.info(f"Found embeddings file: {path}")
                break

        if embeddings_file:
            try:
                # Load pre-computed embeddings
                data = np.load(embeddings_file, allow_pickle=True)
                if isinstance(data, np.lib.npyio.NpzFile):
                    if 'embeddings' in data:
                        stored_embeddings = data['embeddings'].item()
                        logger.info(f"Loaded {len(stored_embeddings)} pre-computed embeddings")

                        # Match images from selected folder with stored embeddings
                        matched_count = 0
                        for image_path in image_paths:
                            image_path_str = str(image_path)
                            image_name = Path(image_path).name

                            # Try exact path match first
                            if image_path_str in stored_embeddings:
                                self.embeddings[image_path_str] = stored_embeddings[image_path_str]
                                matched_count += 1
                            else:
                                # Try filename matching
                                for stored_path, embedding in stored_embeddings.items():
                                    if Path(stored_path).name == image_name:
                                        self.embeddings[image_path_str] = embedding
                                        matched_count += 1
                                        break

                        if matched_count > 0:
                            logger.info(
                                f"Successfully matched {matched_count}/{len(image_paths)} images with embeddings")
                            return
                        else:
                            logger.warning("No matching embeddings found for selected images")

            except Exception as e:
                logger.warning(f"Failed to load pre-computed embeddings: {e}")

        # Generate new embeddings using CLIP if no pre-computed ones found
        logger.info("Generating new CLIP embeddings...")
        self._generate_clip_embeddings(image_paths)

    def _generate_clip_embeddings(self, image_paths):
        """Generate new CLIP embeddings for the images."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch
            from PIL import Image
            from tqdm import tqdm

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLIP model on {device}...")

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            logger.info(f"Generating CLIP embeddings for {len(image_paths)} images...")

            with torch.no_grad():
                for path in tqdm(image_paths, desc="Processing images"):
                    try:
                        image = Image.open(path).convert('RGB')
                        inputs = processor(images=image, return_tensors="pt").to(device)
                        image_features = model.get_image_features(**inputs)
                        embedding = image_features.cpu().numpy()[0]
                        self.embeddings[str(path)] = embedding
                    except Exception as e:
                        logger.warning(f"Error processing {path}: {e}")

            logger.info(f"Generated {len(self.embeddings)} new CLIP embeddings")

            # Optionally save the embeddings
            if self.embeddings:
                try:
                    np.savez("new_embeddings.npz", embeddings=self.embeddings)
                    logger.info("Saved new embeddings to new_embeddings.npz")
                except Exception as e:
                    logger.warning(f"Failed to save embeddings: {e}")

        except ImportError:
            logger.warning("CLIP/transformers not available, using dummy embeddings")
            self._generate_dummy_embeddings(image_paths)
        except Exception as e:
            logger.error(f"Error generating CLIP embeddings: {e}")
            self._generate_dummy_embeddings(image_paths)

    def _generate_dummy_embeddings(self, image_paths):
        """Generate dummy embeddings as fallback."""
        logger.info("Generating dummy embeddings...")
        for path in image_paths:
            self.embeddings[str(path)] = np.random.randn(512)
        logger.info(f"Generated {len(self.embeddings)} dummy embeddings")

    def search_images(self, query, top_k=10, use_optimized_similarity=False):
        """Search images using text query."""
        logger.info(f"Searching for: '{query}' (optimized: {use_optimized_similarity})")

        if not self.embeddings:
            logger.warning("No embeddings available for search")
            return []

        # Generate query embedding
        query_embedding = self._get_query_embedding(query)

        # Calculate similarities
        results = []
        for path, embedding in self.embeddings.items():
            if use_optimized_similarity:
                similarity = self._calculate_optimized_similarity(query_embedding, embedding)
            else:
                # Simple cosine similarity
                similarity = np.dot(query_embedding, embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
                )

            results.append({
                'path': path,
                'score': abs(similarity)  # Use absolute value for ranking
            })

        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _get_query_embedding(self, query):
        """Get embedding for text query."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

            inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)
                return text_features.cpu().numpy()[0]

        except Exception as e:
            logger.warning(f"Error generating query embedding: {e}, using random")
            return np.random.randn(512)

    def _calculate_optimized_similarity(self, query_emb, image_emb):
        """Calculate optimized similarity using searcher parameters."""
        from geometric_metrics import GeometricSimilarityMetrics

        metrics = GeometricSimilarityMetrics()
        return metrics.optimized_similarity(query_emb, image_emb, self.searcher.similarity_params)

    def run_mi_analysis(self, num_pairs=1000, num_bins=20):
        """Run standard MI analysis."""
        if not self.embeddings:
            logger.warning("No embeddings available for MI analysis")
            return None, None

        # Create analyzer with embeddings
        embeddings_list = [(path, emb) for path, emb in self.embeddings.items()]
        analyzer = EnhancedMIAnalysis(embeddings_list, num_pairs, num_bins)
        analyzer.generate_pairs()

        # Run analysis
        mi_results = analyzer.compute_mi_for_all_metrics()

        # Return analyzer and results in expected format
        default_mi = max(mi_results.values()) if mi_results else 0.0
        return analyzer, {'default': default_mi}

    def run_enhanced_mi_analysis(self, num_pairs=1000, num_bins=20, keep_unnormalized=True):
        """Run enhanced MI analysis with multiple geometric metrics."""
        if not self.embeddings:
            logger.warning("No embeddings available for enhanced MI analysis")
            return None, None

        logger.info(f"Running enhanced MI analysis with {len(self.embeddings)} embeddings")

        # Create analyzer with actual data
        embeddings_list = [(path, emb) for path, emb in self.embeddings.items()]

        # Limit number of pairs if we have too many embeddings
        max_pairs = min(num_pairs, 1000)  # Limit to 1000 for performance

        analyzer = EnhancedMIAnalysis(embeddings_list, max_pairs, num_bins, keep_unnormalized)
        analyzer.generate_pairs()  # Back to original method name

        # Run analysis
        logger.info("Computing MI for all metrics...")
        mi_results = analyzer.compute_mi_for_all_metrics()

        logger.info(f"MI analysis complete. Results: {mi_results}")
        return analyzer, mi_results

    def create_mi_visualization(self, filename):
        """Create standard MI visualization."""
        plt.figure(figsize=(8, 6))

        if self.embeddings:
            plt.text(0.5, 0.5,
                     f"Standard MI Analysis\nEmbeddings loaded: {len(self.embeddings)}\nRun analysis to see results",
                     ha='center', va='center', fontsize=12)
        else:
            plt.text(0.5, 0.5, "Standard MI Analysis\nNo embeddings loaded",
                     ha='center', va='center', fontsize=14)

        plt.title("Standard MI Analysis")
        plt.axis('off')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def create_enhanced_mi_visualization(self, filename):
        """Create enhanced MI visualization."""
        plt.figure(figsize=(8, 6))

        if self.embeddings:
            plt.text(0.5, 0.5,
                     f"Enhanced MI Analysis\nEmbeddings loaded: {len(self.embeddings)}\nRun analysis to see results",
                     ha='center', va='center', fontsize=12)
        else:
            plt.text(0.5, 0.5, "Enhanced MI Analysis\nNo embeddings loaded",
                     ha='center', va='center', fontsize=14)

        plt.title("Enhanced MI Analysis")
        plt.axis('off')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        return filename

    def search_with_multiple_metrics(self, query, top_k=5):
        """Search with multiple geometric metrics."""
        logger.info(f"Multi-metric search for: '{query}'")

        if not self.embeddings:
            return {'analysis': {'intersections': {}, 'unique_contributions': {}}}

        # Get query embedding
        query_embedding = self._get_query_embedding(query)

        # Calculate results for different metrics
        from geometric_metrics import GeometricSimilarityMetrics
        metrics_calc = GeometricSimilarityMetrics()

        results_by_metric = {}

        # Cosine similarity
        cosine_results = []
        for path, embedding in self.embeddings.items():
            similarity = metrics_calc.cosine_similarity(query_embedding, embedding)
            cosine_results.append({
                'path': path,
                'cosine_similarity': similarity,
                'score': similarity
            })
        cosine_results.sort(key=lambda x: x['score'], reverse=True)
        results_by_metric['cosine_similarity'] = cosine_results[:top_k]

        # L1 distance (lower is better, so we negate for ranking)
        l1_results = []
        for path, embedding in self.embeddings.items():
            distance = metrics_calc.l1_distance(query_embedding, embedding)
            l1_results.append({
                'path': path,
                'l1_distance': distance,
                'score': -distance  # Negative because lower distance = higher similarity
            })
        l1_results.sort(key=lambda x: x['score'], reverse=True)
        results_by_metric['l1_distance'] = l1_results[:top_k]

        # L2 distance
        l2_results = []
        for path, embedding in self.embeddings.items():
            distance = metrics_calc.l2_distance(query_embedding, embedding)
            l2_results.append({
                'path': path,
                'l2_distance': distance,
                'score': -distance
            })
        l2_results.sort(key=lambda x: x['score'], reverse=True)
        results_by_metric['l2_distance'] = l2_results[:top_k]

        # Calculate intersections
        cosine_paths = set(r['path'] for r in results_by_metric['cosine_similarity'])
        l1_paths = set(r['path'] for r in results_by_metric['l1_distance'])
        l2_paths = set(r['path'] for r in results_by_metric['l2_distance'])

        intersections = {
            'cosine_vs_l1': {
                'intersection_size': len(cosine_paths & l1_paths),
                'intersection_ratio': len(cosine_paths & l1_paths) / top_k if top_k > 0 else 0
            },
            'cosine_vs_l2': {
                'intersection_size': len(cosine_paths & l2_paths),
                'intersection_ratio': len(cosine_paths & l2_paths) / top_k if top_k > 0 else 0
            },
            'l1_vs_l2': {
                'intersection_size': len(l1_paths & l2_paths),
                'intersection_ratio': len(l1_paths & l2_paths) / top_k if top_k > 0 else 0
            }
        }

        # Calculate unique contributions
        all_paths = cosine_paths | l1_paths | l2_paths
        unique_contributions = {
            'cosine_similarity': {
                'unique_count': len(cosine_paths - l1_paths - l2_paths),
                'unique_ratio': len(cosine_paths - l1_paths - l2_paths) / len(all_paths) if all_paths else 0
            },
            'l1_distance': {
                'unique_count': len(l1_paths - cosine_paths - l2_paths),
                'unique_ratio': len(l1_paths - cosine_paths - l2_paths) / len(all_paths) if all_paths else 0
            },
            'l2_distance': {
                'unique_count': len(l2_paths - cosine_paths - l1_paths),
                'unique_ratio': len(l2_paths - cosine_paths - l1_paths) / len(all_paths) if all_paths else 0
            }
        }

        results_by_metric['analysis'] = {
            'intersections': intersections,
            'unique_contributions': unique_contributions
        }

        return results_by_metric


class SimpleSearcher:
    """Simple searcher class for compatibility."""

    def __init__(self):
        self.similarity_params = {
            'w_angle': 1.0,
            'w_l1': 0.0,
            'w_l2': 0.0,
            'w_inf': 0.0,
            'w_mag': 0.0
        }

    def set_similarity_params(self, params):
        """Set similarity parameters."""
        self.similarity_params.update(params)
        logger.info(f"Updated similarity parameters: {self.similarity_params}")


def run_color_analysis(embeddings_file: str, dataset_dir: str, results_dir: str):
    """Run color analysis - compatibility function."""
    try:
        from mi_analysis import analyze_color_embeddings
        return analyze_color_embeddings(embeddings_file, dataset_dir, results_dir)
    except ImportError:
        logger.warning("Color analysis function not available")
        return None


def run_enhanced_mi_analysis(self, num_pairs=1000, num_bins=20, keep_unnormalized=True):
    """Run enhanced MI analysis with COCO categories."""
    if not self.embeddings:
        logger.warning("No embeddings available for enhanced MI analysis")
        return None, None

    logger.info(f"Running enhanced MI analysis with {len(self.embeddings)} embeddings")

    # Create analyzer with actual data
    embeddings_list = [(path, emb) for path, emb in self.embeddings.items()]

    # Limit number of pairs if we have too many embeddings
    max_pairs = min(num_pairs, len(embeddings_list) * (len(embeddings_list) - 1) // 2)

    analyzer = EnhancedMIAnalysis(embeddings_list, max_pairs, num_bins, keep_unnormalized)

    # Use the new COCO-based pair generation
    analyzer.generate_coco_pairs()

    # Run analysis
    logger.info("Computing MI for all metrics...")
    mi_results = analyzer.compute_mi_for_all_metrics()

    logger.info(f"MI analysis complete. Results: {mi_results}")
    return analyzer, mi_results