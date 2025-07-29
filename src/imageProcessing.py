# imageProcessing.py
"""Enhanced image processing utilities for the geometric embedding analysis system."""

import os
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
import logging
from PIL import Image, ImageDraw, ImageEnhance
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
import pandas as pd
from collections import defaultdict

# Try to import optional dependencies
try:
    import cv2

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("OpenCV (cv2) not found. Will use PIL for image processing instead.")

try:
    from sklearn.cluster import KMeans

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not found. Will use simplified color extraction.")

logger = logging.getLogger(__name__)


# Keep your existing functions unchanged
# [Your existing functions]

class ColorDatasetManager:
    """
    Manages the creation and organization of a color-controlled dataset
    for analyzing how different geometric properties of embeddings capture color information.

    This class creates a structured dataset with:
    - 10 object categories × 3 colors × 5 examples per combination
    - Balanced pairs for four relationship types:
      same object/same color, same object/different color,
      different object/same color, different object/different color
    """

    def __init__(self, base_dir: str = "color_dataset"):
        """
        Initialize the dataset manager.

        Args:
            base_dir: Base directory for storing the dataset
        """
        self.base_dir = Path(base_dir)
        self.categories = ["dog", "cat", "horse", "bird", "car",
                           "bottle", "chair", "person", "boat", "bicycle"]
        self.colors = ["brown", "white", "black"]
        self.num_examples = 5  # 5 examples per category-color combination
        self.metadata = []

        # Create directory structure
        os.makedirs(self.base_dir, exist_ok=True)

        for category in self.categories:
            for color in self.colors:
                os.makedirs(self.base_dir / category / color, exist_ok=True)

    def extract_dominant_color(self, image: Image.Image) -> str:
        """
        Extract the dominant color from an image.

        Args:
            image: PIL Image object

        Returns:
            Dominant color name ("brown", "white", "black", or "other")
        """
        if SKLEARN_AVAILABLE:
            # Use k-means for more accurate color extraction
            # Convert to numpy array and reshape for k-means
            img_array = np.array(image)
            pixels = img_array.reshape(-1, 3)

            # Use k-means to find dominant colors (3 clusters)
            kmeans = KMeans(n_clusters=3, n_init=10)
            kmeans.fit(pixels)

            # Get the dominant color (largest cluster)
            colors = kmeans.cluster_centers_.astype(int)
            counts = np.bincount(kmeans.labels_)
            dominant_color = colors[np.argmax(counts)]

            # Classify the color
            r, g, b = dominant_color
        else:
            # Use simple averaging for color extraction
            img_array = np.array(image)
            # Get average color, ignoring very bright pixels (likely background)
            mask = np.max(img_array, axis=2) < 240  # Ignore near-white pixels
            if mask.sum() > 0:  # If we have any non-bright pixels
                dominant_color = np.mean(img_array[mask], axis=0).astype(int)
            else:
                dominant_color = np.mean(img_array, axis=(0, 1)).astype(int)

            r, g, b = dominant_color

        # Simple color classification
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 60 and g < 60 and b < 60:
            return "black"
        elif r > 100 and g < 100 and b < 80:
            return "brown"
        else:
            return "other"

    def filter_coco_images(self, coco_dir: str, annotation_file: str) -> None:
        """
        Filter COCO dataset to find images matching our categories and colors.

        Args:
            coco_dir: Directory containing COCO images
            annotation_file: Path to COCO annotations file
        """
        logger.info("Filtering COCO images by category and color...")

        # Load annotations
        try:
            with open(annotation_file, 'r') as f:
                annotations = json.load(f)
        except Exception as e:
            logger.error(f"Error loading annotation file: {e}")
            return

        # Create mapping from image_id to filename
        id_to_file = {img['id']: img['file_name'] for img in annotations.get('images', [])}
        if not id_to_file:
            logger.error("No image data found in annotations file")
            return

        # Create mapping from image_id to categories
        id_to_categories = defaultdict(list)
        categories_dict = {cat['id']: cat['name'] for cat in annotations.get('categories', [])}
        for ann in annotations.get('annotations', []):
            category_id = ann.get('category_id')
            if category_id:
                category_name = categories_dict.get(category_id)
                if category_name in self.categories:
                    id_to_categories[ann['image_id']].append(category_name)

        # Find suitable images
        selected_images = defaultdict(lambda: defaultdict(list))
        processed_count = 0
        found_count = 0

        for image_id, categories in tqdm(id_to_categories.items(), desc="Analyzing COCO images"):
            if not categories:
                continue

            # Use the first matching category
            category = categories[0]
            processed_count += 1

            # Load image
            try:
                img_path = os.path.join(coco_dir, id_to_file[image_id])
                if not os.path.exists(img_path):
                    logger.warning(f"Image file not found: {img_path}")
                    continue

                image = Image.open(img_path).convert('RGB')

                # Extract color
                color = self.extract_dominant_color(image)

                if color in self.colors:
                    found_count += 1
                    # If we need more images for this category-color combination
                    if len(selected_images[category][color]) < self.num_examples:
                        selected_images[category][color].append((image_id, img_path))
            except Exception as e:
                logger.error(f"Error processing image {image_id}: {str(e)}")

        # Log progress
        logger.info(f"Processed {processed_count} images, found {found_count} matching our criteria")

        # Copy selected images to our dataset
        for category in selected_images:
            for color in selected_images[category]:
                # Create category/color directory
                dest_dir = self.base_dir / category / color
                os.makedirs(dest_dir, exist_ok=True)

                for i, (image_id, img_path) in enumerate(selected_images[category][color]):
                    dest_path = dest_dir / f"{i + 1}.jpg"
                    try:
                        shutil.copy(img_path, dest_path)

                        # Add to metadata
                        self.metadata.append({
                            "path": str(dest_path),
                            "category": category,
                            "color": color,
                            "original_id": image_id,
                            "original_path": img_path
                        })
                    except Exception as e:
                        logger.error(f"Error copying {img_path} to {dest_path}: {e}")

        logger.info(f"Selected {len(self.metadata)} images for the dataset")
        logger.info("COCO image filtering complete.")

    def _draw_shape(self, draw: ImageDraw.Draw, category_idx: int, color: Tuple[int, int, int]) -> None:
        """Draw a simple shape to represent different objects"""
        width, height = 224, 224
        center_x, center_y = width // 2, height // 2

        # Different shapes for different categories
        if category_idx % 10 == 0:  # dog
            # Draw dog-like shape (circle with ears)
            draw.ellipse([center_x - 50, center_y - 50, center_x + 50, center_y + 50], fill=color)
            draw.ellipse([center_x - 70, center_y - 80, center_x - 30, center_y - 40], fill=color)  # left ear
            draw.ellipse([center_x + 30, center_y - 80, center_x + 70, center_y - 40], fill=color)  # right ear

        elif category_idx % 10 == 1:  # cat
            # Draw cat-like shape (circle with pointy ears)
            draw.ellipse([center_x - 40, center_y - 40, center_x + 40, center_y + 40], fill=color)
            draw.polygon([
                (center_x - 40, center_y - 40),
                (center_x - 20, center_y - 80),
                (center_x - 10, center_y - 40)
            ], fill=color)  # left ear
            draw.polygon([
                (center_x + 40, center_y - 40),
                (center_x + 20, center_y - 80),
                (center_x + 10, center_y - 40)
            ], fill=color)  # right ear

        elif category_idx % 10 == 2:  # horse
            # Draw horse-like shape (oval with mane)
            draw.ellipse([center_x - 60, center_y - 30, center_x + 60, center_y + 30], fill=color)
            draw.ellipse([center_x - 70, center_y - 40, center_x - 30, center_y], fill=color)  # head
            draw.rectangle([center_x - 60, center_y - 50, center_x + 30, center_y - 30], fill=color)  # mane

        elif category_idx % 10 == 3:  # bird
            # Draw bird-like shape
            draw.ellipse([center_x - 30, center_y - 20, center_x + 30, center_y + 20], fill=color)  # body
            draw.ellipse([center_x + 20, center_y - 30, center_x + 50, center_y], fill=color)  # head
            draw.polygon([
                (center_x + 40, center_y - 15),
                (center_x + 70, center_y - 20),
                (center_x + 40, center_y)
            ], fill=color)  # beak

        elif category_idx % 10 == 4:  # car
            # Draw car-like shape
            draw.rectangle([center_x - 70, center_y - 20, center_x + 70, center_y + 20], fill=color)  # body
            draw.rectangle([center_x - 50, center_y - 40, center_x + 50, center_y - 20], fill=color)  # top
            draw.ellipse([center_x - 60, center_y + 10, center_x - 30, center_y + 40], fill=color)  # wheel
            draw.ellipse([center_x + 30, center_y + 10, center_x + 60, center_y + 40], fill=color)  # wheel

        else:  # default shape for other categories
            # Draw a simple shape that varies by category index
            shape_type = category_idx % 5

            if shape_type == 0:
                draw.rectangle([center_x - 60, center_y - 60, center_x + 60, center_y + 60], fill=color)
            elif shape_type == 1:
                draw.ellipse([center_x - 60, center_y - 60, center_x + 60, center_y + 60], fill=color)
            elif shape_type == 2:
                draw.polygon([
                    (center_x, center_y - 70),
                    (center_x + 70, center_y + 70),
                    (center_x - 70, center_y + 70)
                ], fill=color)
            elif shape_type == 3:
                draw.rectangle([center_x - 70, center_y - 40, center_x + 70, center_y + 40], fill=color)
            else:
                draw.ellipse([center_x - 70, center_y - 40, center_x + 70, center_y + 40], fill=color)

    def generate_relationship_pairs(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        Generate pairs of images with different relationship types.
        Handles cases with limited data more gracefully.

        Returns:
            Dictionary with relationship types as keys and lists of image pairs as values
        """
        logger.info("Generating relationship pairs...")

        pairs = {
            "same_object_same_color": [],
            "same_object_diff_color": [],
            "diff_object_same_color": [],
            "diff_object_diff_color": []
        }

        # Check if we have enough data
        if len(self.metadata) < 2:
            logger.warning("Not enough images to generate pairs")
            return pairs

        # Group images by category and color
        category_color_images = defaultdict(list)
        for meta in self.metadata:
            key = (meta["category"], meta["color"])
            category_color_images[key].append(meta["path"])

        # For "same_object_same_color" - use images with the same category and color
        for (category, color), paths in category_color_images.items():
            if len(paths) >= 2:  # Need at least 2 images
                for i in range(len(paths)):
                    for j in range(i + 1, len(paths)):
                        pairs["same_object_same_color"].append((paths[i], paths[j]))

        # For "same_object_diff_color" - use different colors of the same object category
        for category in self.categories:
            category_colors = [color for (cat, color), paths in category_color_images.items()
                               if cat == category and paths]

            if len(category_colors) >= 2:  # Need at least 2 colors
                for color1_idx, color1 in enumerate(category_colors):
                    for color2 in category_colors[color1_idx + 1:]:
                        paths1 = category_color_images[(category, color1)]
                        paths2 = category_color_images[(category, color2)]

                        # Create pairs using all combinations
                        for path1 in paths1:
                            for path2 in paths2:
                                pairs["same_object_diff_color"].append((path1, path2))

        # For "diff_object_same_color" - use same color, different object categories
        for color in self.colors:
            color_categories = [cat for (cat, col), paths in category_color_images.items()
                                if col == color and paths]

            if len(color_categories) >= 2:  # Need at least 2 categories
                for cat1_idx, cat1 in enumerate(color_categories):
                    for cat2 in color_categories[cat1_idx + 1:]:
                        paths1 = category_color_images[(cat1, color)]
                        paths2 = category_color_images[(cat2, color)]

                        # Create pairs using all combinations
                        for path1 in paths1:
                            for path2 in paths2:
                                pairs["diff_object_same_color"].append((path1, path2))

        # For "diff_object_diff_color" - different categories with different colors
        categories_with_images = set(cat for (cat, _), paths in category_color_images.items() if paths)

        if len(categories_with_images) >= 2:  # Need at least 2 categories
            category_list = list(categories_with_images)
            for cat1_idx, cat1 in enumerate(category_list):
                for cat2 in category_list[cat1_idx + 1:]:
                    # Get all colors for each category
                    colors1 = [col for (c, col), paths in category_color_images.items()
                               if c == cat1 and paths]
                    colors2 = [col for (c, col), paths in category_color_images.items()
                               if c == cat2 and paths]

                    # For each color combination
                    for color1 in colors1:
                        for color2 in colors2:
                            if color1 != color2:  # Ensure different colors
                                paths1 = category_color_images[(cat1, color1)]
                                paths2 = category_color_images[(cat2, color2)]

                                # Create pairs using all combinations
                                for path1 in paths1:
                                    for path2 in paths2:
                                        pairs["diff_object_diff_color"].append((path1, path2))

        # Balance pairs to have similar counts if possible
        for rel_type in list(pairs.keys()):
            if not pairs[rel_type]:
                logger.warning(f"No pairs found for relationship type: {rel_type}")

        # Log the number of pairs
        for rel_type, rel_pairs in pairs.items():
            logger.info(f"Generated {len(rel_pairs)} {rel_type} pairs")

        return pairs

    def save_metadata(self) -> None:
        """Save dataset metadata to CSV file"""
        df = pd.DataFrame(self.metadata)
        metadata_path = self.base_dir / "metadata.csv"
        df.to_csv(metadata_path, index=False)
        logger.info(f"Metadata saved to {metadata_path}")

    def create_dataset(self, coco_dir: Optional[str] = None,
                       annotation_file: Optional[str] = None) -> Dict[str, List[Tuple[str, str]]]:
        """
        Create the complete dataset, either from COCO or from scratch.

        Args:
            coco_dir: Optional, directory containing COCO images
            annotation_file: Optional, path to COCO annotations file

        Returns:
            Dictionary with relationship types as keys and lists of image pairs as values
        """
        if coco_dir and annotation_file:
            # Use COCO dataset as source
            self.filter_coco_images(coco_dir, annotation_file)
        else:
            # Create synthetic images from scratch
            logger.info("COCO dataset not provided. Using synthetic dataset only.")
            self._create_synthetic_dataset_from_scratch()

        # Create synthetic variations to ensure balance
        self.create_synthetic_variations()

        # Save metadata
        self.save_metadata()

        # Generate relationship pairs
        pairs = self.generate_relationship_pairs()

        # Save pairs to file for later use
        pairs_path = self.base_dir / "pairs.json"
        with open(pairs_path, 'w') as f:
            # Convert paths to be relative to base_dir for portability
            base_str = str(self.base_dir) + os.sep
            serializable_pairs = {}
            for relation_type, relation_pairs in pairs.items():
                serializable_pairs[relation_type] = [
                    (p1.replace(base_str, '') if p1.startswith(base_str) else p1,
                     p2.replace(base_str, '') if p2.startswith(base_str) else p2)
                    for p1, p2 in relation_pairs
                ]
            json.dump(serializable_pairs, f)

        logger.info(f"Dataset creation complete. Data stored in {self.base_dir}")
        return pairs

    def visualize_dataset_examples(self, output_path: Optional[str] = None) -> None:
        """
        Create a visualization of dataset examples for the thesis.

        Shows examples of each relationship type to illustrate the dataset structure.

        Args:
            output_path: Optional path to save the visualization. If None, displays it.
        """
        try:
            import matplotlib.pyplot as plt

            # Load relationship pairs
            pairs_path = self.base_dir / "pairs.json"
            if not pairs_path.exists():
                logger.error("Pairs file not found. Run create_dataset first.")
                return

            with open(pairs_path, 'r') as f:
                pairs = json.load(f)

            # Create figure
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))

            # Set up row titles
            row_titles = [
                "Same Object, Same Color",
                "Same Object, Different Color",
                "Different Object, Same Color",
                "Different Object, Different Color"
            ]

            for i, relation_type in enumerate(pairs.keys()):
                # Get 2 examples for this relationship
                example_pairs = pairs[relation_type][:2]

                for j, (rel_path1, rel_path2) in enumerate(example_pairs):
                    # Load images (with proper path handling)
                    path1 = os.path.join(self.base_dir, rel_path1)
                    path2 = os.path.join(self.base_dir, rel_path2)

                    img1 = Image.open(path1).convert('RGB')
                    img2 = Image.open(path2).convert('RGB')

                    # Display images
                    axes[i, j * 2].imshow(img1)
                    axes[i, j * 2 + 1].imshow(img2)

                    # Remove axes
                    axes[i, j * 2].axis('off')
                    axes[i, j * 2 + 1].axis('off')

                # Add row title
                axes[i, 0].set_ylabel(row_titles[i], fontsize=12)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=300)
                logger.info(f"Dataset visualization saved to {output_path}")
            else:
                plt.show()
        except ImportError:
            logger.error("Matplotlib not found. Visualization skipped.")


def prepare_color_dataset(coco_dir: str = "/home/lena/Desktop/COCO_images/val2017",
                          annotation_file: str = None,
                          base_dir: str = "color_dataset") -> Tuple[
    Dict[str, List[Tuple[str, str]]], List[Dict[str, Any]]]:
    """
    Prepare a color-controlled dataset for analyzing how CLIP embeddings capture color information
    using the COCO dataset.

    Args:
        coco_dir: Directory containing COCO images
        annotation_file: Path to COCO annotations file (required)
        base_dir: Directory to store the dataset

    Returns:
        Tuple of (pairs, metadata) where:
        - pairs is a dictionary with relationship types as keys and lists of image pairs as values
        - metadata is a list of dictionaries with information about each image
    """
    # Validate inputs
    if not os.path.isdir(coco_dir):
        logger.error(f"COCO directory {coco_dir} not found. Please specify a valid directory.")
        return {}, []

    if annotation_file is None or not os.path.isfile(annotation_file):
        logger.error(f"COCO annotation file not provided or not found. This is required for dataset creation.")
        return {}, []

    # Create the dataset manager
    dataset_manager = ColorDatasetManager(base_dir)

    # Process COCO images directly
    logger.info(f"Using COCO dataset at {coco_dir}")
    dataset_manager.filter_coco_images(coco_dir, annotation_file)

    # Save metadata
    dataset_manager.save_metadata()

    # Generate relationship pairs
    pairs = dataset_manager.generate_relationship_pairs()

    # Save pairs to file
    pairs_path = Path(base_dir) / "pairs.json"
    if pairs:
        with open(pairs_path, 'w') as f:
            # Convert paths to be relative to base_dir for portability
            base_str = str(base_dir) + os.sep
            serializable_pairs = {}
            for relation_type, relation_pairs in pairs.items():
                serializable_pairs[relation_type] = [
                    (p1.replace(base_str, '') if p1.startswith(base_str) else p1,
                     p2.replace(base_str, '') if p2.startswith(base_str) else p2)
                    for p1, p2 in relation_pairs
                ]
            json.dump(serializable_pairs, f)

    # Generate visualization if we have enough data
    if len(dataset_manager.metadata) > 0:
        dataset_manager.visualize_dataset_examples(os.path.join(base_dir, "dataset_examples.png"))

    logger.info(f"Dataset creation complete. Found {len(dataset_manager.metadata)} images.")
    logger.info(f"Generated pairs: {', '.join([f'{len(p)} {r}' for r, p in pairs.items()])}")

    # Return the pairs and metadata for analysis
    return pairs, dataset_manager.metadata