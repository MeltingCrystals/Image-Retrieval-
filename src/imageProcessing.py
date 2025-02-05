import os
import shutil
from pathlib import Path
from typing import List

def get_photo_directory():
    # Create a directory for sample images
    photo_dir = Path("/home/lena/IWB_photos")

    # Verify the directory exists
    if not photo_dir.exists():
        raise FileNotFoundError(f"Directory {photo_dir} does not exist")

    # Get list of jpg files
    jpg_files = list(photo_dir.glob("*.jpg"))
    print(f"Found {len(jpg_files)} jpg files in {photo_dir}")


    return photo_dir, jpg_files


def scan_image_directory(directory_path: str = "/home/lena/IWB_photos") -> List[Path]:
    # Supported image extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    image_paths = []
    directory = Path(directory_path)

    # Check if directory exists
    if not directory.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")

    # Walk through directory recursively
    for file_path in directory.rglob('*'):
        if file_path.suffix.lower() in valid_extensions:
            image_paths.append(file_path)

    return image_paths


