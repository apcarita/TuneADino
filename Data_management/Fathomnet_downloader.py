"""
Downloads ALL 212,965+ images from FathomNet for self-supervised learning.
Optimized for maximum speed and efficiency.

"""

import os
import sys
import time
import requests
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from typing import List, Optional, Dict, Any
from tqdm import tqdm
import json
from io import BytesIO
from PIL import Image

try:
    from fathomnet.api import images
    from fathomnet.dto import Pageable
    FATHOMNET_AVAILABLE = True
except ImportError:
    FATHOMNET_AVAILABLE = False
    print("ERROR: FathomNet API not available! Install with: pip install fathomnet")
    sys.exit(1)


class FathomNetMaxDownloader:
    def __init__(self, output_dir: str = "Fathomnet_whole", max_workers: int = 30, 
                 target_width: int = 1080, target_height: int = 720):
        self.output_dir = Path(output_dir)
        self.max_workers = max_workers
        self.target_width = target_width
        self.target_height = target_height
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistics
        self.downloaded = 0
        self.failed = 0
        self.invalid_images = 0
        self.skipped = 0
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'FathomNet-Max-Downloader/1.0'
        })
        
        # No resume capability - keep it simple
        
    
    def get_total_count(self) -> int:
        """Get total number of images available."""
        try:
            count = images.count_all()
            return count.count if hasattr(count, 'count') else count
        except Exception as e:
            print(f"Could not get total count: {e}")
            return 0
    
    def fetch_all_images(self) -> List[Dict[str, Any]]:
        """Fetch ALL images from FathomNet - really all of them."""
        try:
            print(f"[DEBUG] Fetching ALL images from FathomNet...")
            
            # Get total count first
            total_count = self.get_total_count()
            print(f"[DEBUG] Total images available: {total_count}")
            
            # Use a very large page size to get everything
            # FathomNet has ~212k images, so use 250k to be safe
            large_pageable = Pageable(page=0, size=250000)
            all_images = images.find_all(pageable=large_pageable)
            
            if not all_images:
                print(f"[WARNING] API returned no images")
                return []
            
            print(f"[DEBUG] Successfully fetched {len(all_images)} total images")
            
            # Convert to simple dictionaries - no filtering, download everything
            batch_data = []
            for img in all_images:
                img_dict = {
                    'uuid': getattr(img, 'uuid', ''),
                    'url': getattr(img, 'url', ''),
                    'width': getattr(img, 'width', None),
                    'height': getattr(img, 'height', None),
                }
                batch_data.append(img_dict)
            
            return batch_data
            
        except Exception as e:
            print(f"[ERROR] CRITICAL: Failed to fetch images: {type(e).__name__}: {e}")
            print(f"[ERROR] This is a serious API failure - stopping download")
            raise  # Re-raise the exception to stop execution
    
    def get_filename(self, url: str, uuid: str) -> str:
        """Generate filename from URL and UUID - use full UUID for uniqueness."""
        parsed_url = urlparse(url)
        original_name = Path(parsed_url.path).name
        
        if original_name and '.' in original_name:
            name, ext = os.path.splitext(original_name)
            # Use full UUID to prevent collisions
            return f"{uuid}_{name}{ext}"
        else:
            return f"{uuid}.jpg"
    
    def download_image(self, image_data: Dict[str, Any]) -> Optional[str]:
        """Download, validate, resize and save a single image."""
        url = image_data.get('url', '')
        uuid = image_data.get('uuid', '')
        
        if not url or not uuid:
            return "Missing URL or UUID"
        
        filename = self.get_filename(url, uuid)
        # Force .jpg extension for consistency
        if not filename.lower().endswith('.jpg'):
            filename = os.path.splitext(filename)[0] + '.jpg'
        filepath = self.output_dir / filename
        
        # Skip if file already exists and is valid
        if filepath.exists():
            try:
                # Quick validation check
                with Image.open(filepath) as img:
                    # Verify it's the right size (means it was processed correctly)
                    if img.size == (self.target_width, self.target_height):
                        self.skipped += 1
                        return None
            except:
                # File exists but is corrupt, re-download it
                pass
        
        try:
            # Download image data
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # Quick content type check
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                self.failed += 1
                return f"Not an image: {filename}"
            
            # Validate and process image
            try:
                # Load image from bytes
                img = Image.open(BytesIO(response.content))
                
                # Verify it's a valid image by loading it
                img.verify()
                
                # Reopen for processing (verify() closes the file)
                img = Image.open(BytesIO(response.content))
                
                # Convert to RGB (handles RGBA, grayscale, etc.)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to target dimensions
                img = img.resize((self.target_width, self.target_height), Image.Resampling.LANCZOS)
                
                # Save as JPEG with good quality
                img.save(filepath, 'JPEG', quality=85, optimize=True)
                
                self.downloaded += 1
                return None
                
            except Exception as img_error:
                self.invalid_images += 1
                return f"Invalid/corrupt image {filename}: {type(img_error).__name__}"
            
        except Exception as e:
            self.failed += 1
            error_msg = f"Error downloading {filename}: {type(e).__name__}: {str(e)[:100]}"
            # Log critical network errors more prominently
            if any(keyword in str(e).lower() for keyword in ['timeout', 'connection', 'network', 'dns']):
                print(f"[NETWORK ERROR] {error_msg}")
            return error_msg
    
    def download_all_images(self, all_images: List[Dict[str, Any]]) -> None:
        """Download all images with a progress bar."""
        if not all_images:
            return
        
        print(f"Starting download of {len(all_images)} images...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.download_image, img_data)
                for img_data in all_images
            ]
            
            # Simple progress bar for all downloads
            pbar = tqdm(as_completed(futures), total=len(all_images), desc="Downloading", unit="img")
            for future in pbar:
                result = future.result()
                if result:  # Error message
                    tqdm.write(f"[ERROR] {result}")
    
    def run_max_download(self, limit: Optional[int] = None, auto_yes: bool = False):
        """Download all images - simple and straightforward."""
        print("=== FathomNet Maximum Image Downloader ===")
        print("Downloading ALL FathomNet images for AI training")
        print(f"Images will be resized to {self.target_width}x{self.target_height} and validated")
        print()
        
        print(f"Max concurrent downloads: {self.max_workers}")
        print(f"Output directory: {self.output_dir.absolute()}")
        print()
        
        # Confirm
        if not auto_yes:
            response = input(f"Download ALL FathomNet images? This will use significant bandwidth. (y/n): ")
            if not response.lower().startswith('y'):
                print("Download cancelled.")
                return
        else:
            print(f"Auto-confirm enabled: proceeding to download all images.")
        
        start_time = time.time()
        
        # Fetch all images at once
        try:
            all_images = self.fetch_all_images()
        except Exception as e:
            print(f"\n[ERROR] FATAL: Failed to fetch images")
            print(f"[ERROR] Exception: {type(e).__name__}: {e}")
            raise
        
        if not all_images:
            print("No images found!")
            return
        
        # Apply limit if specified
        if limit:
            all_images = all_images[:limit]
            print(f"Limited to first {limit} images")
        
        print(f"Found {len(all_images)} images to download")
        
        # Download all images
        self.download_all_images(all_images)
        
        # Final stats
        elapsed = time.time() - start_time
        total_processed = self.downloaded + self.skipped
        print(f"\n=== Download Complete ===")
        print(f"Downloaded: {self.downloaded:,}")
        print(f"Skipped (already exists): {self.skipped:,}")
        print(f"Failed: {self.failed:,}")
        print(f"Invalid/Corrupt: {self.invalid_images:,}")
        print(f"Total processed: {total_processed:,}")
        print(f"Total time: {elapsed/3600:.1f} hours")
        if self.downloaded > 0:
            print(f"Average download rate: {self.downloaded/elapsed:.1f} images/second")
        print(f"Image size: {self.target_width}x{self.target_height}")
        print(f"Output: {self.output_dir.absolute()}")


def main():
    parser = argparse.ArgumentParser(description="FathomNet max image downloader.")
    parser.add_argument("--output-dir", default="Fathomnet_whole", help="Directory to save images.")
    parser.add_argument("--workers", type=int, default=30, help="Number of concurrent download threads.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of images to download (for testing).")
    parser.add_argument("--auto-yes", action="store_true", help="Bypass the confirmation prompt.")
    args = parser.parse_args()
    
    print("=== Configuration ===")
    print(f"Output directory: {args.output_dir}")
    print(f"Max workers: {args.workers}")
    print(f"Limit: {args.limit if args.limit else 'None (download ALL)'}")
    print()
    
    downloader = FathomNetMaxDownloader(
        output_dir=args.output_dir,
        max_workers=args.workers
    )
    
    try:
        downloader.run_max_download(limit=args.limit, auto_yes=args.auto_yes)
    except KeyboardInterrupt:
        print("\nDownload interrupted. Progress saved - you can resume later!")
    except Exception as e:
        print(f"Error: {e}")
        


if __name__ == "__main__":
    main()
