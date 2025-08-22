#!/usr/bin/env python3

import os
import subprocess
import sys

def test_videos_simple():
    """Simple test of videos with decord"""
    data_dir = "/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"
    
    test_videos = [
        "kids_videos/a_child_sitting_on_the_floor_playing_with_a_laptop.mp4",
        "kids_videos/a_baby_laying_on_a_hospital_bed_while_a_nurse_holds_her.mp4", 
        "0_30_eufy_videos/0211.mp4",
        "0_30_eufy_videos/1168.mp4",
        "smarthome/smartbench_0001.mp4"
    ]
    
    good_videos = []
    hevc_videos = []
    
    print("Testing videos for HEVC SPS errors:")
    print("=" * 50)
    
    for video in test_videos:
        video_path = os.path.join(data_dir, video)
        
        if not os.path.exists(video_path):
            print(f"❌ {video}: Not found")
            continue
            
        print(f"\n🎬 {video}")
        
        # Test with decord - capture both stdout and stderr
        test_code = f'''
import decord
try:
    vr = decord.VideoReader("{video_path}")
    frames = len(vr)
    fps = vr.get_avg_fps()
    print("SUCCESS", frames, fps)
except Exception as e:
    print("ERROR", str(e))
'''
        
        # Run the test and capture stderr
        result = subprocess.run([
            '/home/ec2-user/SageMaker/efs/conda_envs/trl/bin/python3', '-c', test_code
        ], capture_output=True, text=True, cwd=data_dir)
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        print(f"   stdout: {stdout}")
        
        if stdout.startswith("SUCCESS"):
            parts = stdout.split()
            frames = parts[1]
            fps = parts[2]
            print(f"   ✅ Valid: {frames} frames, {fps} fps")
            
            # Check for HEVC errors in stderr
            if "SPS 0 does not exist" in stderr:
                print(f"   ⚠️  Has HEVC SPS errors")
                hevc_videos.append(video)
            else:
                print(f"   🟢 No HEVC errors")
                good_videos.append(video)
                
        else:
            print(f"   ❌ Failed to load")
            
        if stderr:
            hevc_count = stderr.count("SPS 0 does not exist")
            if hevc_count > 0:
                print(f"   📊 HEVC SPS errors: {hevc_count}")
    
    print(f"\n" + "=" * 50)
    print(f"RESULTS:")
    print(f"Good videos (no HEVC): {len(good_videos)}")
    for v in good_videos:
        print(f"  ✅ {v}")
        
    print(f"Videos with HEVC errors: {len(hevc_videos)}")
    for v in hevc_videos:
        print(f"  ⚠️  {v}")
        
    return good_videos, hevc_videos

if __name__ == "__main__":
    test_videos_simple()