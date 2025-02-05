# this file will coordinate everything
from pathlib import Path
from typing import List
import tkinter as tk
from tkinter import filedialog
from ImageEmbeddingSystem import ImageEmbeddingSystem
from image_search import TextImageSearcher  # We'll create this


class ImageSearchApp:
    def __init__(self):
        self.embedding_system = ImageEmbeddingSystem()
        self.searcher = TextImageSearcher()

    def scan_folders(self) -> List[Path]:
        """Open dialog for folder selection and return image paths"""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        folder_path = filedialog.askdirectory(title="Select Folder with Images")

        if folder_path:
            folder = Path(folder_path)
            image_paths = list(folder.glob("*.jpg"))
            return image_paths
        return []

    def process_images(self, image_paths: List[Path]):
        """Process images and store embeddings"""
        if not image_paths:
            return
        self.embedding_system.process_and_store_images(image_paths)

    def search_images(self, text_query: str, top_k: int = 5):
        """Search for images matching text query"""
        return self.searcher.search(text_query, top_k)