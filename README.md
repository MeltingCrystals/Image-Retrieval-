# Image Retrieval Project

## Description
Machine learning project for image retrieval with color-based analysis using various distance metrics and embedding techniques.

## Project Structure
```
src/
├── ImageEmbeddingSystem.py           # Main embedding system
├── app_pipeline.py                  # Application pipeline  
├── config.py                       # Configuration settings
├── geometric_metrics.py            # Distance metric implementations
├── imageProcessing.py              # Image processing utilities
├── image_search.py                 # Search functionality
├── main.py                         # Main execution script
├── mi_analysis.py                  # Mutual information analysis
├── color_analysis_workflow.py     # Color analysis workflow
├── test_basic.py                   # Basic tests
└── color_analysis_results/
    └── analysis_results/           # Analysis plots and metrics
        ├── precision_recall_curves.png
        ├── color_metrics_comparison.png
        ├── distance_density_plots.png
        ├── summary.png
        └── results.json
```

## Setup Instructions

### 1. Clone and Setup Environment
```bash
git clone https://github.com/YOUR_USERNAME/ImageRetrieval.git
cd ImageRetrieval
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Running the Project
```bash
cd src
python main.py
```

## Features
- **Multi-metric similarity**: L1, L2, L∞, cosine distance
- **Color analysis**: Color-based image retrieval evaluation
- **Embedding systems**: Support for various image embedding models
- **Evaluation metrics**: Precision-recall analysis and performance comparison

## Results Included
This repository contains comprehensive analysis results:
- **Precision-recall curves** for different distance metrics
- **Distance distribution analysis** (L1, L2, L∞, cosine, angle)
- **Color metrics comparison** visualizations
- **Performance summary** charts and quantitative metrics
- **Mutual information analysis** results

## Requirements
- Python 3.10+
- PyTorch with CUDA support
- OpenCV, NumPy, Matplotlib
- Milvus Lite for vector database operations

## Dataset Notes
The original image dataset is excluded due to size constraints. All analysis results and visualizations are preserved, demonstrating the system's performance across different similarity metrics.
