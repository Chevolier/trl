#!/usr/bin/env python3

import json
import os
import sys
from pathlib import Path

def test_videos(data_dir="/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"):
    """Test some videos to find good ones and ones with HEVC errors"""
    
    # Test videos from kids_videos first
    test_videos = [
        "kids_videos/a_child_sitting_on_the_floor_playing_with_a_laptop.mp4",
        "kids_videos/a_baby_laying_on_a_hospital_bed_while_a_nurse_holds_her.mp4", 
        "kids_videos/a_baby_looking_away_and_the_hands_of_its_mother.mp4",
        "0_30_eufy_videos/0211.mp4",
        "0_30_eufy_videos/1168.mp4",
        "smarthome/smartbench_0001.mp4",
        "smarthome/smartbench_0002.mp4",
        "12_actionData_videos/output1_A2_L2_2_ffmpeg.mp4_12.mp4"
    ]
    
    good_videos = []
    hevc_error_videos = []
    
    print("Testing videos for HEVC errors and validity:")
    print("=" * 60)
    
    for video in test_videos:
        video_path = os.path.join(data_dir, video)
        
        if not os.path.exists(video_path):
            print(f"❌ {video}: File not found")
            continue
            
        file_size = os.path.getsize(video_path)
        print(f"\n🎬 Testing: {video}")
        print(f"   File size: {file_size} bytes")
        
        try:
            import decord
            
            # Capture stderr to detect HEVC warnings
            import subprocess
            import tempfile
            
            # Test with decord and capture any stderr output
            with tempfile.NamedTemporaryFile(mode='w+') as stderr_file:
                # Run a simple decord test in a subprocess to capture stderr
                test_code = f"""
import decord
import sys
try:
    vr = decord.VideoReader('{video_path}')
    frames = len(vr)
    fps = vr.get_avg_fps()
    print(f"SUCCESS:{frames}:{fps}")
except Exception as e:
    print(f"ERROR:{e}")
"""
                
                result = subprocess.run([
                    sys.executable, '-c', test_code
                ], capture_output=True, text=True)
                
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                
                if stdout.startswith("SUCCESS"):
                    _, frames, fps = stdout.split(":")
                    frames = int(frames)
                    fps = float(fps)
                    
                    has_hevc_error = "SPS 0 does not exist" in stderr
                    
                    print(f"   ✅ Valid: {frames} frames, {fps:.2f} fps")
                    if has_hevc_error:
                        print(f"   ⚠️  Has HEVC SPS errors in stderr")
                        hevc_error_videos.append(video)
                    else:
                        print(f"   🟢 Clean (no HEVC errors)")
                        good_videos.append(video)
                        
                elif stdout.startswith("ERROR"):
                    error = stdout[6:]  # Remove "ERROR:" prefix
                    print(f"   ❌ Decord error: {error}")
                else:
                    print(f"   ❌ Unexpected output: {stdout}")
                    
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    print(f"\n" + "=" * 60)
    print(f"📊 SUMMARY:")
    print(f"   Good videos (no HEVC errors): {len(good_videos)}")
    for v in good_videos:
        print(f"      ✅ {v}")
        
    print(f"   Videos with HEVC errors: {len(hevc_error_videos)}")
    for v in hevc_error_videos:
        print(f"      ⚠️  {v}")
    
    return good_videos, hevc_error_videos

def create_test_dataset(good_videos, hevc_error_videos, data_dir):
    """Create a small test dataset with both good and problematic videos"""
    
    # Load original data to get the conversation structure
    kids_file = os.path.join(data_dir, "kids_videos.json")
    with open(kids_file, 'r') as f:
        kids_data = json.load(f)
    
    # Create test dataset with mix of good and bad videos
    test_items = []
    
    # Add some good videos
    for i, video in enumerate(good_videos[:3]):
        if i < len(kids_data):
            item = kids_data[i].copy()
            item['videos'] = [video]
            test_items.append(item)
    
    # Add some HEVC error videos  
    for i, video in enumerate(hevc_error_videos[:2]):
        if i + 3 < len(kids_data):
            item = kids_data[i + 3].copy()
            item['videos'] = [video]
            test_items.append(item)
    
    # Add one non-existent video
    if len(kids_data) > 5:
        item = kids_data[5].copy()
        item['videos'] = ["non_existent/fake_video.mp4"]
        test_items.append(item)
    
    # Save test dataset
    test_file = os.path.join(data_dir, "test_videos.json")
    with open(test_file, 'w') as f:
        json.dump(test_items, f, indent=2)
    
    print(f"\n📄 Created test dataset: {test_file}")
    print(f"   Total items: {len(test_items)}")
    print(f"   - Good videos: {len(good_videos[:3])}")
    print(f"   - HEVC error videos: {len(hevc_error_videos[:2])}")
    print(f"   - Non-existent videos: 1")
    
    return test_file

def test_filter_script(test_file, data_dir):
    """Test the filter script with our test dataset"""
    print(f"\n🔧 Testing filter script...")
    
    # Modify the filter script to work with our test file
    import subprocess
    
    # Create a modified version of the filter script for testing
    filter_code = f"""
import sys
sys.path.append('.')
from filter_video_datasets import filter_video_datasets
import json
import os

# Test with our specific file
json_files = ["test_videos.json"]
data_dir = "{data_dir}"

for json_file in json_files:
    file_path = os.path.join(data_dir, json_file)
    if not os.path.exists(file_path):
        print(f"Test file not found: {{file_path}}")
        continue
        
    print(f"Testing with {{json_file}}...")
    
    # Load original
    with open(file_path, 'r') as f:
        annotations = json.load(f)
    
    original_count = len(annotations)
    print(f"  Original items: {{original_count}}")
    
    # Process items (simplified version)
    valid_items = []
    corrupt_items = []
    
    for i, item in enumerate(annotations):
        videos = item.get('videos', [])
        if not videos:
            corrupt_items.append(item)
            continue
            
        all_videos_valid = True
        invalid_reasons = []
        
        for video in videos:
            video_path = os.path.join(data_dir, video)
            
            if not os.path.exists(video_path):
                all_videos_valid = False
                invalid_reasons.append(f"not_found: {{video}}")
                continue
                
            file_size = os.path.getsize(video_path)
            if file_size <= 1024:
                all_videos_valid = False
                invalid_reasons.append(f"too_small: {{video}}")
                continue
                
            try:
                import decord
                vr = decord.VideoReader(video_path)
                total_frames = len(vr)
                if total_frames <= 0:
                    all_videos_valid = False
                    invalid_reasons.append(f"no_frames: {{video}}")
                else:
                    print(f"    ✅ Valid: {{os.path.basename(video)}} ({{total_frames}} frames)")
            except Exception as e:
                all_videos_valid = False
                invalid_reasons.append(f"decord_error: {{video}}")
                print(f"    ❌ Error: {{os.path.basename(video)}} - {{e}}")
        
        if all_videos_valid:
            valid_items.append(item)
        else:
            corrupt_item = item.copy()
            corrupt_item['_corruption_reason'] = invalid_reasons
            corrupt_items.append(corrupt_item)
            print(f"    ⚠️  Corrupt: {{invalid_reasons}}")
    
    print(f"  ✅ Valid items: {{len(valid_items)}}")
    print(f"  ❌ Corrupt items: {{len(corrupt_items)}}")
    print(f"  📊 Success rate: {{len(valid_items)/original_count*100:.1f}}%")
"""
    
    result = subprocess.run([
        '/home/ec2-user/SageMaker/efs/conda_envs/trl/bin/python3', '-c', filter_code
    ], capture_output=True, text=True)
    
    print("Filter test output:")
    print(result.stdout)
    if result.stderr:
        print("Errors:")
        print(result.stderr)

if __name__ == "__main__":
    data_dir = "/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"
    
    # Step 1: Test videos to categorize them
    good_videos, hevc_error_videos = test_videos(data_dir)
    
    # Step 2: Create test dataset
    test_file = create_test_dataset(good_videos, hevc_error_videos, data_dir)
    
    # Step 3: Test the filter script
    test_filter_script(test_file, data_dir)