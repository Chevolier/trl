#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path

def find_hevc_error_videos(data_dir="/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video", max_videos_to_test=200):
    """Search through datasets to find videos that produce HEVC SPS errors"""
    
    json_files = [
        "0_30_eufy_videos.json",
        "12_actionData_videos.json", 
        "kids_videos.json",
        "smarthome_videos_2.json",
        "videos_val_oops.json"
    ]
    
    hevc_error_videos = []
    good_videos = []
    tested_count = 0
    
    print("🔍 Searching for videos with HEVC SPS errors...")
    print("=" * 60)
    
    for json_file in json_files:
        if tested_count >= max_videos_to_test:
            break
            
        file_path = os.path.join(data_dir, json_file)
        if not os.path.exists(file_path):
            print(f"❌ {json_file}: file not found")
            continue
            
        print(f"\n📂 Checking {json_file}...")
        
        # Load data
        with open(file_path, 'r') as f:
            annotations = json.load(f)
        
        for i, item in enumerate(annotations):
            if tested_count >= max_videos_to_test:
                break
                
            videos = item.get('videos', [])
            if not videos:
                continue
                
            for video in videos:
                if tested_count >= max_videos_to_test:
                    break
                    
                video_path = os.path.join(data_dir, video)
                
                if not os.path.exists(video_path):
                    continue
                
                tested_count += 1
                
                # Test video processing to capture HEVC errors
                test_code = f'''
import decord
import sys

try:
    # Try to read the video with decord
    vr = decord.VideoReader("{video_path}")
    frames = len(vr)
    fps = vr.get_avg_fps()
    
    # Also try to get some frames to trigger more processing
    if frames > 0:
        # Get first few frames to trigger decoding
        sample_indices = list(range(min(5, frames)))
        frame_data = vr.get_batch(sample_indices)
    
    print(f"SUCCESS {{frames}} {{fps}}")
except Exception as e:
    print(f"ERROR {{str(e)}}")
'''
                
                # Run test and capture stderr for HEVC errors
                result = subprocess.run([
                    '/home/ec2-user/SageMaker/efs/conda_envs/trl/bin/python3', '-c', test_code
                ], capture_output=True, text=True, timeout=10)
                
                stdout = result.stdout.strip()
                stderr = result.stderr.strip()
                
                # Count HEVC SPS errors
                hevc_error_count = stderr.count("SPS 0 does not exist")
                
                if hevc_error_count > 0:
                    hevc_error_videos.append({
                        'video': video,
                        'video_path': video_path,
                        'dataset': json_file,
                        'hevc_errors': hevc_error_count,
                        'stdout': stdout,
                        'can_decode': stdout.startswith("SUCCESS")
                    })
                    print(f"   ⚠️  FOUND HEVC: {os.path.basename(video)} ({hevc_error_count} SPS errors)")
                    
                    # Stop if we found 5 HEVC error videos
                    if len(hevc_error_videos) >= 5:
                        print(f"   🎯 Found target of 5 HEVC error videos!")
                        break
                        
                elif stdout.startswith("SUCCESS"):
                    if len(good_videos) < 5:  # Keep some good videos for comparison
                        good_videos.append({
                            'video': video,
                            'video_path': video_path,
                            'dataset': json_file,
                            'stdout': stdout
                        })
                
                # Progress update
                if tested_count % 20 == 0:
                    print(f"   📊 Tested: {tested_count}, HEVC found: {len(hevc_error_videos)}, Good: {len(good_videos)}")
            
            if len(hevc_error_videos) >= 5:
                break
                
        if len(hevc_error_videos) >= 5:
            break
    
    print(f"\n" + "=" * 60)
    print(f"🔍 SEARCH RESULTS")
    print(f"=" * 60)
    print(f"Videos tested: {tested_count}")
    print(f"HEVC error videos found: {len(hevc_error_videos)}")
    print(f"Good videos found: {len(good_videos)}")
    
    print(f"\n⚠️  HEVC ERROR VIDEOS:")
    for i, video_info in enumerate(hevc_error_videos, 1):
        print(f"{i}. {video_info['video']}")
        print(f"   Dataset: {video_info['dataset']}")
        print(f"   HEVC errors: {video_info['hevc_errors']}")
        print(f"   Can decode: {video_info['can_decode']}")
        print(f"   Status: {video_info['stdout']}")
    
    print(f"\n✅ GOOD VIDEOS (for comparison):")
    for i, video_info in enumerate(good_videos[:3], 1):
        print(f"{i}. {video_info['video']}")
        print(f"   Dataset: {video_info['dataset']}")
        print(f"   Status: {video_info['stdout']}")
    
    return hevc_error_videos, good_videos

def create_hevc_test_dataset(hevc_videos, good_videos, data_dir):
    """Create a test dataset with HEVC error videos and good videos"""
    
    # Load a sample structure from existing data
    kids_file = os.path.join(data_dir, "kids_videos.json")
    with open(kids_file, 'r') as f:
        sample_data = json.load(f)
    
    test_items = []
    
    # Add HEVC error videos
    for i, video_info in enumerate(hevc_videos):
        if i < len(sample_data):
            item = sample_data[i].copy()
            item['videos'] = [video_info['video']]
            item['_test_type'] = 'hevc_error'
            item['_expected_hevc_errors'] = video_info['hevc_errors']
            test_items.append(item)
    
    # Add good videos
    for i, video_info in enumerate(good_videos):
        if i + len(hevc_videos) < len(sample_data):
            item = sample_data[i + len(hevc_videos)].copy()
            item['videos'] = [video_info['video']]
            item['_test_type'] = 'good_video'
            test_items.append(item)
    
    # Add a non-existent video
    if len(test_items) < len(sample_data):
        item = sample_data[len(test_items)].copy()
        item['videos'] = ["test/nonexistent_video.mp4"]
        item['_test_type'] = 'missing_file'
        test_items.append(item)
    
    # Save test dataset
    test_file = os.path.join(data_dir, "hevc_test_videos.json")
    with open(test_file, 'w') as f:
        json.dump(test_items, f, indent=2)
    
    print(f"\n📄 Created HEVC test dataset: {test_file}")
    print(f"   Total items: {len(test_items)}")
    print(f"   - HEVC error videos: {len(hevc_videos)}")
    print(f"   - Good videos: {len(good_videos)}")
    print(f"   - Missing files: 1")
    
    return test_file

def test_filter_with_hevc_videos(test_file, data_dir):
    """Test our filter script specifically on videos with HEVC errors"""
    
    print(f"\n🧪 Testing filter script on HEVC error videos...")
    
    # Create test script as a temporary file to avoid complex string formatting
    import tempfile
    
    test_script_content = f"""import json
import os
import decord
import subprocess
import sys

data_dir = "{data_dir}"
test_file = "{test_file}"

print("Loading HEVC test dataset...")
with open(test_file, 'r') as f:
    data = json.load(f)

print(f"Test items: {{len(data)}}")

valid_items = []
corrupt_items = []
hevc_detected = []

for i, item in enumerate(data):
    video_path_rel = item['videos'][0]
    video_path = os.path.join(data_dir, video_path_rel)
    test_type = item.get('_test_type', 'unknown')
    
    print(f"\\n📋 Item {{i+1}} ({{test_type}}): {{os.path.basename(video_path_rel)}}")
    
    # Our filter logic
    videos = item.get('videos', [])
    all_valid = True
    issues = []
    
    for video in videos:
        video_full_path = os.path.join(data_dir, video)
        
        # Check existence
        if not os.path.exists(video_full_path):
            all_valid = False
            issues.append(f"not_found: {{video}}")
            print(f"   ❌ File not found")
            continue
        
        # Check size
        size = os.path.getsize(video_full_path)
        if size < 1024:
            all_valid = False
            issues.append(f"too_small: {{video}}")
            print(f"   ❌ Too small: {{size}} bytes")
            continue
            
        # Check with decord - this is where we might catch HEVC issues
        try:
            # Suppress stderr to avoid flooding output
            old_stderr = os.dup(2)
            os.close(2)
            os.open(os.devnull, os.O_RDWR)
            
            try:
                vr = decord.VideoReader(video_full_path)
                frames = len(vr)
                
                # Try to actually decode some frames
                if frames > 0:
                    sample_indices = list(range(min(3, frames)))
                    frame_data = vr.get_batch(sample_indices)
                    
            finally:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
            
            if frames > 0:
                print(f"   ✅ Decord OK: {{frames}} frames, {{size}} bytes")
                
                # Separately test for HEVC errors in stderr
                hevc_test_code = '''import decord
vr = decord.VideoReader("''' + video_full_path + '''")
frames = len(vr)
if frames > 0:
    sample_indices = list(range(min(3, frames)))
    frame_data = vr.get_batch(sample_indices)
print("DECODE_SUCCESS")'''
                
                result = subprocess.run([
                    '/home/ec2-user/SageMaker/efs/conda_envs/trl/bin/python3', '-c', hevc_test_code
                ], capture_output=True, text=True, timeout=5)
                
                hevc_error_count = result.stderr.count("SPS 0 does not exist")
                if hevc_error_count > 0:
                    print(f"   ⚠️  HEVC SPS errors detected: {{hevc_error_count}}")
                    hevc_detected.append({{
                        'video': video,
                        'hevc_errors': hevc_error_count,
                        'test_type': test_type
                    }})
                
            else:
                all_valid = False
                issues.append(f"no_frames: {{video}}")
                print(f"   ❌ No frames")
        except Exception as e:
            all_valid = False
            issues.append(f"decord_error: {{video}}")
            print(f"   ❌ Decord error: {{e}}")
    
    # Classify result
    if all_valid:
        valid_items.append(item)
        print(f"   → VALID (filter accepts)")
    else:
        item_copy = item.copy()
        item_copy['_corruption_reason'] = issues
        corrupt_items.append(item_copy)
        print(f"   → CORRUPT (filter rejects): {{issues}}")

print(f"\\n📊 FILTER TEST RESULTS:")
print(f"   Original items: {{len(data)}}")
print(f"   Filter accepts: {{len(valid_items)}}")
print(f"   Filter rejects: {{len(corrupt_items)}}")
print(f"   HEVC errors detected: {{len(hevc_detected)}}")

# Analyze by test type
type_results = {{}}
for item in data:
    test_type = item.get('_test_type', 'unknown')
    type_results[test_type] = type_results.get(test_type, {{'total': 0, 'valid': 0, 'corrupt': 0}})
    type_results[test_type]['total'] += 1

for item in valid_items:
    test_type = item.get('_test_type', 'unknown')
    type_results[test_type]['valid'] += 1
    
for item in corrupt_items:
    test_type = item.get('_test_type', 'unknown')
    type_results[test_type]['corrupt'] += 1

print(f"\\n📊 Results by type:")
for test_type, counts in type_results.items():
    total = counts['total']
    valid = counts['valid']
    corrupt = counts['corrupt']
    print(f"   {{test_type}}: {{valid}}/{{total}} valid, {{corrupt}}/{{total}} corrupt")

# Show HEVC detection details
if hevc_detected:
    print(f"\\n⚠️  HEVC errors found in:")
    for item in hevc_detected:
        print(f"   {{item['video']}} ({{item['test_type']}}): {{item['hevc_errors']}} errors")
else:
    print(f"\\n🟢 No HEVC errors detected in processed videos")
"""
    
    # Write test script to temp file and run it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as tmp:
        tmp.write(test_script_content)
        tmp.flush()
        
        result = subprocess.run([
            '/home/ec2-user/SageMaker/efs/conda_envs/trl/bin/python3', tmp.name
        ], capture_output=True, text=True)
    
    # Clean up temp file
    os.unlink(tmp.name)
    
    print("Filter test results:")
    print(result.stdout)
    if result.stderr:
        print("Stderr:")
        print(result.stderr[:1000])  # Truncate if too long

if __name__ == "__main__":
    data_dir = "/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"
    
    # Step 1: Search for HEVC error videos
    hevc_videos, good_videos = find_hevc_error_videos(data_dir, max_videos_to_test=500)
    
    if len(hevc_videos) == 0:
        print("❌ No HEVC error videos found! Trying larger sample...")
        hevc_videos, good_videos = find_hevc_error_videos(data_dir, max_videos_to_test=1000)
    
    if len(hevc_videos) > 0:
        # Step 2: Create test dataset with HEVC videos
        test_file = create_hevc_test_dataset(hevc_videos, good_videos, data_dir)
        
        # Step 3: Test filter on HEVC videos
        test_filter_with_hevc_videos(test_file, data_dir)
    else:
        print("❌ Could not find any videos with HEVC SPS errors in the tested sample")
        print("🔍 The videos in this dataset may not have HEVC corruption issues")