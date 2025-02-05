from pathlib import Path
from app_pipeline import ImageSearchApp


def main():
    # Initialize app
    app = ImageSearchApp()

    # Scan for images
    print("Please select a folder with images to process...")
    image_paths = app.scan_folders()

    if not image_paths:
        print("No valid images found in the selected folder. Exiting...")
        return

    print(f"Found {len(image_paths)} images. Processing...")
    app.process_images(image_paths)
    print("Processing complete!")

    # Example search
    while True:
        query = input("\nEnter text to search for images (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        results = app.search_images(query)
        if not results:
            print("No matching images found with sufficient confidence.")
        else:
            print("\nMatching images:")
            for idx, match in enumerate(results, 1):
                # Get just the filename from the full path
                filename = Path(match['path']).name
                confidence = match['score'] * 100  # Convert to percentage
                print(f"{idx}. {filename}")
                print(f"   Confidence: {confidence:.1f}%")
                print(f"   Full path: {match['path']}")
                print()

if __name__ == "__main__":
    main()
