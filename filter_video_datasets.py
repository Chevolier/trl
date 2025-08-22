#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Suppress HEVC warnings
import warnings
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

def filter_video_datasets(data_dir="/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"):
    """Filter video datasets to keep only items with valid videos that can be read by decord"""
    
    json_files = [
        "0_30_eufy_videos.json",
        "12_actionData_videos.json", 
        "kids_videos.json",
        "smarthome_videos_2.json",
        "videos_val_oops.json"
    ]
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        if not os.path.exists(file_path):
            print(f"Skipping {json_file}: file not found")
            continue
            
        print(f"\nProcessing {json_file}...")
        
        # Load the JSON data
        with open(file_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        original_count = len(annotations)
        print(f"  Original items: {original_count}")
        
        valid_items = []
        corrupt_items = []
        
        # Process items with progress bar
        for i, item in enumerate(tqdm(annotations, desc=f"  Processing {json_file}", leave=False)):
            try:
                # Extract video paths
                videos = item.get('videos', [])
                if not videos:
                    corrupt_items.append(item)
                    continue
                
                # Check each video
                all_videos_valid = True
                invalid_videos = []
                
                for video in videos:
                    video_path = os.path.join(data_dir, video)
                    
                    # Check if file exists
                    if not os.path.exists(video_path):
                        all_videos_valid = False
                        invalid_videos.append(f"not_found: {video}")
                        continue
                    
                    # Check file size
                    file_size = os.path.getsize(video_path)
                    if file_size <= 1024:  # Less than 1KB
                        all_videos_valid = False
                        invalid_videos.append(f"too_small: {video}")
                        continue
                    
                    # Try to read with decord
                    try:
                        import decord
                        # Suppress stderr to avoid HEVC warnings
                        old_stderr = os.dup(2)
                        os.close(2)
                        os.open(os.devnull, os.O_RDWR)
                        try:
                            vr = decord.VideoReader(video_path)
                            total_frames = len(vr)
                        finally:
                            os.dup2(old_stderr, 2)
                            os.close(old_stderr)
                        
                        if total_frames <= 0:
                            all_videos_valid = False
                            invalid_videos.append(f"no_frames: {video}")
                    except Exception as e:
                        all_videos_valid = False
                        invalid_videos.append(f"decord_error: {video}")
                
                # Add item to appropriate list
                if all_videos_valid:
                    valid_items.append(item)
                else:
                    # Add corruption info to the item
                    corrupt_item = item.copy()
                    corrupt_item['_corruption_reason'] = invalid_videos
                    corrupt_items.append(corrupt_item)
                    
            except Exception as e:
                corrupt_item = item.copy()
                corrupt_item['_corruption_reason'] = [f"processing_error: {e}"]
                corrupt_items.append(corrupt_item)
        
        # Save filtered dataset
        base_name = json_file.replace('.json', '')
        filtered_file = os.path.join(data_dir, f"{base_name}_filtered.json")
        corrupt_file = os.path.join(data_dir, f"{base_name}_corrupt.json")
        
        # Write filtered items
        with open(filtered_file, 'w', encoding='utf-8') as f:
            json.dump(valid_items, f, indent=2, ensure_ascii=False)
        
        # Write corrupt items
        with open(corrupt_file, 'w', encoding='utf-8') as f:
            json.dump(corrupt_items, f, indent=2, ensure_ascii=False)
        
        filtered_count = len(valid_items)
        corrupt_count = len(corrupt_items)
        
        print(f"  Filtered items: {filtered_count}")
        print(f"  Corrupt items: {corrupt_count}")
        print(f"  Success rate: {filtered_count/original_count*100:.1f}%")
        print(f"  Saved to: {filtered_file}")
        print(f"  Corrupt saved to: {corrupt_file}")

if __name__ == "__main__":
    filter_video_datasets()