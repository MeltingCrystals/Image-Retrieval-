#!/usr/bin/env python3
# color_analysis_workflow.py
"""
Complete workflow for color-based analysis of CLIP embeddings.

This script:
1. Creates the color dataset using imageProcessing.py
2. Generates CLIP embeddings for the dataset
3. Runs the geometric information theory analysis
4. Displays key results and visualizations
"""

import os
import sys
import argparse
from pathlib import Path
import logging
import numpy as np
import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import modules - ensure they're in your Python path
from imageProcessing import prepare_color_dataset
from transformers import CLIPModel, CLIPProcessor
from app_pipeline import EnhancedImageSearchApp, run_color_analysis


def main():
    parser = argparse.ArgumentParser(description="Color-based analysis of CLIP embeddings")
    parser.add_argument("--coco_dir", required=True,
                        help="Path to COCO dataset (required)")
    parser.add_argument("--annotation_file", required=True,
                        help="Path to COCO annotations (required)")
    parser.add_argument("--output_dir", default="color_analysis", help="Output directory")
    parser.add_argument("--skip_dataset", action="store_true",
                        help="Skip dataset creation and use existing dataset")
    parser.add_argument("--skip_embeddings", action="store_true",
                        help="Skip embedding generation and use existing embeddings")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Prepare the color dataset
    dataset_dir = os.path.join(args.output_dir, "color_dataset")
    if args.skip_dataset and os.path.exists(dataset_dir):
        logger.info("Using existing dataset at {}".format(dataset_dir))
        # Load metadata
        import pandas as pd

        metadata_path = os.path.join(dataset_dir, "metadata.csv")
        if os.path.exists(metadata_path):
            metadata = pd.read_csv(metadata_path)
            logger.info(f"Loaded metadata for {len(metadata)} images")
        else:
            logger.warning("Metadata file not found. Cannot proceed with existing dataset.")
            return
    else:
        logger.info("=== Step 1: Preparing Color Dataset ===")
        pairs, metadata = prepare_color_dataset(
            coco_dir=args.coco_dir,
            annotation_file=args.annotation_file,
            base_dir=dataset_dir
        )

        if not metadata:
            logger.error("Failed to create dataset. Check your COCO dataset path and annotation file.")
            return

        logger.info(f"Created color dataset with {len(metadata)} images")
        if pairs:
            logger.info(f"Generated {len(next(iter(pairs.values())))} pairs per relationship type")
        logger.info(f"Dataset visualization saved to {dataset_dir}/dataset_examples.png")
    # Step 2: Generate CLIP embeddings
    embeddings_file = os.path.join(args.output_dir, "color_embeddings.npz")
    if args.skip_embeddings and os.path.exists(embeddings_file):
        logger.info(f"Using existing embeddings at {embeddings_file}")
    else:
        logger.info("=== Step 2: Generating CLIP Embeddings ===")

        # Initialize CLIP model and processor
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Process all images and generate embeddings
        all_image_paths = []
        if isinstance(metadata, list):
            # It's a list of dictionaries
            for item in metadata:
                path = item["path"]
                # Ensure path is properly resolved without duplication
                if os.path.isabs(path):
                    all_image_paths.append(path)
                else:
                    # Check if path already starts with dataset_dir
                    if path.startswith(dataset_dir):
                        all_image_paths.append(path)
                    else:
                        all_image_paths.append(os.path.join(dataset_dir, path))
        else:
            # It's a DataFrame
            for _, row in metadata.iterrows():
                path = row["path"]
                # Ensure path is properly resolved without duplication
                if os.path.isabs(path):
                    all_image_paths.append(path)
                else:
                    # Check if path already starts with dataset_dir
                    if path.startswith(dataset_dir):
                        all_image_paths.append(path)
                    else:
                        all_image_paths.append(os.path.join(dataset_dir, path))
        logger.info(f"Processing {len(all_image_paths)} images...")
        # Generate embeddings
        embeddings = {}

        with torch.no_grad():
            for path in tqdm(all_image_paths, desc="Generating embeddings"):
                try:
                    # Open image and process
                    from PIL import Image
                    image = Image.open(path).convert('RGB')
                    inputs = processor(images=image, return_tensors="pt").to(device)

                    # Get image features
                    image_features = model.get_image_features(**inputs)
                    embedding = image_features.cpu().numpy()[0]

                    # Store embedding
                    embeddings[path] = embedding
                except Exception as e:
                    logger.warning(f"Error processing image {path}: {e}")

        # Save embeddings to file
        np.savez(embeddings_file, embeddings=embeddings)
        logger.info(f"Saved embeddings for {len(embeddings)} images to {embeddings_file}")

    # Step 3: Run the geometric information theory analysis
    logger.info("=== Step 3: Running Geometric Information Theory Analysis ===")
    results_dir = os.path.join(args.output_dir, "analysis_results")
    run_color_analysis(
        embeddings_file=embeddings_file,
        dataset_dir=dataset_dir,
        results_dir=results_dir
    )

    logger.info("=== Analysis Complete! ===")
    logger.info(f"All results saved to {args.output_dir}")
    logger.info("Key files:")
    logger.info(f"- Dataset: {dataset_dir}")
    logger.info(f"- Embeddings: {embeddings_file}")
    logger.info(f"- Analysis results: {results_dir}")
    logger.info(f"- Summary visualization: {results_dir}/summary.png")

    # Display some key results
    results_json = os.path.join(results_dir, "results.json")
    if os.path.exists(results_json):
        import json
        with open(results_json, 'r') as f:
            results = json.load(f)

        # Print color-specific MI values
        logger.info("\nColor-specific Mutual Information:")
        color_mi = results.get("color_mi", {})
        for metric, mi_value in sorted(color_mi.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {metric}: {mi_value:.4f} bits")

        # Print optimal weights
        logger.info("\nOptimal weights for similarity function:")
        optimal_weights = results.get("optimal_weights", {})
        for param, weight in optimal_weights.items():
            if weight > 0.01:  # Only show non-zero weights
                logger.info(f"  {param}: {weight:.2f}")

        # Calculate improvement over standard cosine similarity
        if "cosine_distance" in color_mi:
            cosine_mi = color_mi["cosine_distance"]
            best_metric, best_mi = max(color_mi.items(), key=lambda x: x[1])
            improvement = ((best_mi - cosine_mi) / cosine_mi) * 100 if cosine_mi > 0 else float('inf')
            logger.info(f"\nBest metric: {best_metric} with {best_mi:.4f} bits")
            logger.info(f"Improvement over cosine similarity: {improvement:.1f}%")


if __name__ == "__main__":
    main()
