#!/usr/bin/env python3

import os
import subprocess
import sys

def test_qwen_processing():
    """Test video processing through qwen_vl_utils like in training"""
    data_dir = "/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"
    
    test_videos = [
        "kids_videos/a_child_sitting_on_the_floor_playing_with_a_laptop.mp4",
        "0_30_eufy_videos/0211.mp4",
        "smarthome/smartbench_0001.mp4"
    ]
    
    print("Testing video processing with qwen_vl_utils:")
    print("=" * 60)
    
    for video in test_videos:
        video_path = os.path.join(data_dir, video)
        
        if not os.path.exists(video_path):
            print(f"❌ {video}: Not found")
            continue
            
        print(f"\n🎬 {video}")
        
        # Test with qwen_vl_utils like in the actual training
        test_code = f'''
from qwen_vl_utils import process_vision_info

# Create sample prompt with video (like in training)
prompts = [[
    {{
        "role": "user", 
        "content": [
            {{
                "type": "video", 
                "video": "{video_path}", 
                "fps": 1, 
                "max_frames": 16
            }},
            {{
                "type": "text", 
                "text": "Describe this video"
            }}
        ]
    }}
]]

print("Processing with qwen_vl_utils...")
try:
    images, videos, video_kwargs = process_vision_info(prompts, return_video_kwargs=True)
    print(f"SUCCESS videos_shape={{videos[0].shape if videos else None}}")
except Exception as e:
    print(f"ERROR {{str(e)}}")
'''
        
        # Run the test and capture stderr to see HEVC errors
        result = subprocess.run([
            '/home/ec2-user/SageMaker/efs/conda_envs/trl/bin/python3', '-c', test_code
        ], capture_output=True, text=True, cwd=data_dir)
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        print(f"   stdout: {stdout}")
        
        if stderr:
            hevc_count = stderr.count("SPS 0 does not exist")
            if hevc_count > 0:
                print(f"   ⚠️  HEVC SPS errors: {hevc_count}")
                print(f"   stderr (first 200 chars): {stderr[:200]}...")
            else:
                print(f"   Other stderr: {stderr[:100]}")
        else:
            print(f"   🟢 No errors in stderr")

def create_test_dataset():
    """Create a small test dataset for filter testing"""
    
    import json
    
    data_dir = "/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"
    
    # Create a mini dataset with known good videos and some problematic cases
    test_data = [
        {
            "conversations": [
                {"from": "human", "value": "Describe the video"},
                {"from": "gpt", "value": "This is a test video"}
            ],
            "videos": ["kids_videos/a_child_sitting_on_the_floor_playing_with_a_laptop.mp4"]
        },
        {
            "conversations": [
                {"from": "human", "value": "What do you see?"},
                {"from": "gpt", "value": "Another test video"}
            ],
            "videos": ["0_30_eufy_videos/0211.mp4"]
        },
        {
            "conversations": [
                {"from": "human", "value": "Analyze this"},
                {"from": "gpt", "value": "Third test video"}
            ],
            "videos": ["smarthome/smartbench_0001.mp4"]
        },
        {
            "conversations": [
                {"from": "human", "value": "What happens here?"},
                {"from": "gpt", "value": "Missing video test"}
            ],
            "videos": ["missing/nonexistent_video.mp4"]
        }
    ]
    
    # Save test dataset
    test_file = os.path.join(data_dir, "mini_test_videos.json")
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"\n📄 Created mini test dataset: {test_file}")
    print(f"   Items: {len(test_data)}")
    print(f"   - 3 existing videos")
    print(f"   - 1 missing video")
    
    return test_file

def test_filter_on_mini_dataset(test_file):
    """Test our filter script on the mini dataset"""
    
    print(f"\n🔧 Testing filter script on mini dataset...")
    
    # Create a simple test of the filter logic
    filter_test = f'''
import json
import os
import decord

data_dir = "/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"
test_file = "{test_file}"

print("Loading test dataset...")
with open(test_file, 'r') as f:
    data = json.load(f)

print(f"Original items: {{len(data)}}")

valid_items = []
corrupt_items = []

for i, item in enumerate(data):
    print(f"\\n📋 Item {{i+1}}: {{item['videos'][0]}}")
    
    videos = item.get('videos', [])
    all_valid = True
    issues = []
    
    for video in videos:
        video_path = os.path.join(data_dir, video)
        
        # Check existence
        if not os.path.exists(video_path):
            all_valid = False
            issues.append(f"not_found: {{video}}")
            print(f"   ❌ File not found")
            continue
        
        # Check size
        size = os.path.getsize(video_path)
        if size < 1024:
            all_valid = False
            issues.append(f"too_small: {{video}}")
            print(f"   ❌ Too small: {{size}} bytes")
            continue
            
        # Check with decord
        try:
            vr = decord.VideoReader(video_path)
            frames = len(vr)
            if frames > 0:
                print(f"   ✅ Valid: {{frames}} frames, {{size}} bytes")
            else:
                all_valid = False
                issues.append(f"no_frames: {{video}}")
                print(f"   ❌ No frames")
        except Exception as e:
            all_valid = False
            issues.append(f"decord_error: {{video}}")
            print(f"   ❌ Decord error: {{e}}")
    
    if all_valid:
        valid_items.append(item)
        print(f"   → VALID")
    else:
        item_copy = item.copy()
        item_copy['_corruption_reason'] = issues
        corrupt_items.append(item_copy)
        print(f"   → CORRUPT: {{issues}}")

print(f"\\n📊 FILTER RESULTS:")
print(f"   Original: {{len(data)}}")
print(f"   Valid: {{len(valid_items)}}")
print(f"   Corrupt: {{len(corrupt_items)}}")
print(f"   Success rate: {{len(valid_items)/len(data)*100:.1f}}%")

# Save results
base_name = test_file.replace('.json', '')
filtered_file = base_name + '_filtered.json'
corrupt_file = base_name + '_corrupt.json'

with open(filtered_file, 'w') as f:
    json.dump(valid_items, f, indent=2)
    
with open(corrupt_file, 'w') as f:
    json.dump(corrupt_items, f, indent=2)
    
print(f"   ✅ Saved filtered: {{filtered_file}}")
print(f"   ❌ Saved corrupt: {{corrupt_file}}")
'''
    
    result = subprocess.run([
        '/home/ec2-user/SageMaker/efs/conda_envs/trl/bin/python3', '-c', filter_test
    ], capture_output=True, text=True)
    
    print("Filter test results:")
    print(result.stdout)
    if result.stderr:
        print("Stderr:")
        print(result.stderr)

if __name__ == "__main__":
    # Test qwen processing
    test_qwen_processing()
    
    # Create mini dataset
    test_file = create_test_dataset()
    
    # Test filter
    test_filter_on_mini_dataset(test_file)