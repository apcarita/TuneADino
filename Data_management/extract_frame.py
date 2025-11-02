#!/usr/bin/env python3
"""
Extract frames from videos in WebUOT-1M dataset structure.
Extracts 1 frame per second from each video and saves to output folder.
"""

import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse


class VideoFrameExtractor:
    def __init__(self, input_dir, output_dir, progress_file='extraction_progress.json', limit=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.progress_file = progress_file
        self.limit = limit
        self.progress = self.load_progress()
        
    def load_progress(self):
        """Load progress from JSON file if it exists."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'processed_videos': [], 'current_index': 0}
    
    def save_progress(self):
        """Save current progress to JSON file."""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def find_all_videos(self):
        """Find all .mp4 files in the dataset structure."""
        videos = []
        for video_path in sorted(self.input_dir.rglob('*.mp4')):
            videos.append(video_path)
        return videos
    
    def extract_frames_from_video(self, video_path):
        """Extract frames from a video at 1 fps."""
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"\nError: Could not open video {video_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0:
            print(f"\nError: Invalid FPS for video {video_path}")
            cap.release()
            return False
        
        # Calculate frame interval for 1 fps extraction
        frame_interval = int(fps)
        
        # Get video identifier from parent folder name
        video_name = video_path.parent.name
        
        frame_count = 0
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at 1 fps intervals - all in one directory with video name prefix
            if frame_count % frame_interval == 0:
                frame_filename = self.output_dir / f"{video_name}_frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved_count += 1
            
            frame_count += 1
        
        cap.release()
        return True
    
    def process_all_videos(self):
        """Process all videos with progress tracking and resume capability."""
        videos = self.find_all_videos()
        
        if not videos:
            print(f"No videos found in {self.input_dir}")
            return
        
        # Apply limit if specified
        if self.limit:
            videos = videos[:self.limit]
        
        # Filter out already processed videos
        videos_to_process = [
            v for v in videos 
            if str(v) not in self.progress['processed_videos']
        ]
        
        if not videos_to_process:
            print("All videos already processed!")
            return
        
        print(f"Found {len(videos)} total videos")
        print(f"Already processed: {len(self.progress['processed_videos'])}")
        print(f"Remaining to process: {len(videos_to_process)}")
        print(f"Output directory: {self.output_dir}")
        print("-" * 60)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process videos with progress bar
        for video_path in tqdm(videos_to_process, desc="Extracting frames"):
            try:
                success = self.extract_frames_from_video(video_path)
                
                if success:
                    # Update progress
                    self.progress['processed_videos'].append(str(video_path))
                    self.progress['current_index'] = len(self.progress['processed_videos'])
                    self.save_progress()
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Progress has been saved.")
                print(f"Resume by running the script again.")
                self.save_progress()
                break
            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                continue
        
        print(f"\n\nExtraction complete!")
        print(f"Total videos processed: {len(self.progress['processed_videos'])}")
        print(f"Frames saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from WebUOT-1M dataset videos at 1 fps'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing video folders (e.g., /path/to/Test)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='extracted_frames',
        help='Output directory for extracted frames (default: extracted_frames)'
    )
    parser.add_argument(
        '--progress-file',
        type=str,
        default='extraction_progress.json',
        help='Progress file for resume capability (default: extraction_progress.json)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of videos to process (for testing)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset progress and start from beginning'
    )
    
    args = parser.parse_args()
    
    # Reset progress if requested
    if args.reset and os.path.exists(args.progress_file):
        os.remove(args.progress_file)
        print("Progress reset.")
    
    # Create extractor and process videos
    extractor = VideoFrameExtractor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        progress_file=args.progress_file,
        limit=args.limit
    )
    
    extractor.process_all_videos()


if __name__ == '__main__':
    main()

