#!/usr/bin/env python3
"""
Image Resizer for Large Datasets
Resizes all JPG and PNG images in a directory to 1920x1080 to save storage.
Optimized for handling large datasets (50k+ images).
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor
import argparse
from tqdm import tqdm


def resize_image(image_path, target_size=(1920, 1080), quality=85, delete_original=True):
    """
    Resize a single image to target size and optionally delete original.
    
    Args:
        image_path: Path to the image file
        target_size: Tuple of (width, height)
        quality: JPEG quality (1-100)
        delete_original: Whether to delete original after successful resize
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        original_size = os.path.getsize(image_path)
        
        # Determine output format and paths
        is_png = image_path.suffix.lower() == '.png'
        if is_png:
            # Convert PNG to JPEG
            final_path = image_path.with_suffix('.jpg')
            temp_path = image_path.with_suffix('.jpg.tmp')
        else:
            # Keep as JPEG
            final_path = image_path
            temp_path = image_path.with_suffix(f"{image_path.suffix}.tmp")
        
        with Image.open(image_path) as img:
            # Convert RGBA to RGB for JPEG compatibility
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Resize image maintaining aspect ratio and crop to exact size
            img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
            
            # Save to temp file with optimization (always JPEG now for compression)
            img.save(temp_path, 'JPEG', quality=quality, optimize=True)
        
        # Verify temp file was created successfully
        if not os.path.exists(temp_path):
            return False, f"Failed to create temp file: {image_path.name}"
        
        new_size = os.path.getsize(temp_path)
        if new_size == 0:
            os.remove(temp_path)
            return False, f"Temp file is empty: {image_path.name}"
        
        # Replace original with resized version
        os.replace(temp_path, final_path)
        
        # If we converted PNG to JPG, remove the original PNG
        if is_png and final_path != image_path:
            os.remove(image_path)
        
        if delete_original:
            size_saved = original_size - new_size
            size_saved_mb = size_saved / (1024 * 1024)
            format_change = " (PNG→JPG)" if is_png else ""
            return True, f"Resized & saved {size_saved_mb:.1f}MB{format_change}: {final_path.name}"
        else:
            return True, f"Resized: {final_path.name}"
    
    except Exception as e:
        # Clean up temp file if it exists
        temp_path = image_path.with_suffix(f"{image_path.suffix}.tmp")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass
        return False, f"Error processing {image_path.name}: {str(e)}"


def get_image_files(directory):
    """Get all JPG and PNG files from directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    extensions = {'.jpg', '.jpeg', '.png'}
    image_files = []
    
    for ext in extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    return image_files


def resize_images_batch(directory, target_size=(1920, 1080), quality=85, max_workers=4, delete_original=True):
    """
    Resize all images in a directory using multithreading.
    
    Args:
        directory: Path to directory containing images
        target_size: Tuple of (width, height)
        quality: JPEG quality (1-100)
        max_workers: Number of threads for parallel processing
        delete_original: Whether to delete originals after successful resize
    """
    print(f"Scanning directory: {directory}")
    image_files = get_image_files(directory)
    
    if not image_files:
        print("No JPG or PNG files found in the directory.")
        return
    
    print(f"Found {len(image_files)} images to process")
    if delete_original:
        print("WARNING: Original images will be deleted after successful resize!")
        response = input("Continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Operation cancelled.")
            return
    
    success_count = 0
    error_count = 0
    total_saved_mb = 0.0
    
    # Process images with progress bar
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        desc = "Resizing & deleting" if delete_original else "Resizing images"
        with tqdm(total=len(image_files), desc=desc, unit="img") as pbar:
            futures = []
            
            for image_path in image_files:
                future = executor.submit(resize_image, image_path, target_size, quality, delete_original)
                futures.append(future)
            
            for future in futures:
                success, message = future.result()
                if success:
                    success_count += 1
                    # Extract saved MB from message if available
                    if "saved" in message and "MB:" in message:
                        try:
                            saved_mb = float(message.split("saved ")[1].split("MB:")[0])
                            total_saved_mb += saved_mb
                        except:
                            pass
                else:
                    error_count += 1
                    print(f"\n{message}")
                
                pbar.update(1)
                postfix = {'Success': success_count, 'Errors': error_count}
                if delete_original and total_saved_mb > 0:
                    postfix['Saved'] = f"{total_saved_mb:.1f}MB"
                pbar.set_postfix(postfix)
    
    print(f"\nProcessing complete!")
    print(f"Successfully resized: {success_count} images")
    print(f"Errors: {error_count} images")
    if delete_original and total_saved_mb > 0:
        print(f"Total storage saved: {total_saved_mb:.1f}MB ({total_saved_mb/1024:.1f}GB)")


def main():
    parser = argparse.ArgumentParser(
        description="Resize all JPG and PNG images in a directory to 1920x1080"
    )
    parser.add_argument(
        "directory", 
        help="Directory containing images to resize"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=1920, 
        help="Target width (default: 1920)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=1080, 
        help="Target height (default: 1080)"
    )
    parser.add_argument(
        "--quality", 
        type=int, 
        default=85, 
        help="JPEG quality 1-100 (default: 85)"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4, 
        help="Number of worker threads (default: 4)"
    )
    parser.add_argument(
        "--keep-original", 
        action="store_true", 
        help="Keep original images (don't delete after resize)"
    )
    
    args = parser.parse_args()
    
    if not (1 <= args.quality <= 100):
        print("Quality must be between 1 and 100")
        sys.exit(1)
    
    target_size = (args.width, args.height)
    delete_original = not args.keep_original
    
    try:
        resize_images_batch(
            args.directory, 
            target_size, 
            args.quality, 
            args.workers,
            delete_original
        )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
