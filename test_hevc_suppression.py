#!/usr/bin/env python3

import decord
# Set decord's log level to suppress warnings (QUIET suppresses all but critical)
decord.logging.set_level(decord.logging.QUIET)  # QUIET = -8

# Test with one of the HEVC error videos we found earlier
video_path = "/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video/0_30_eufy_videos/2312.mp4"

print("Testing HEVC SPS error suppression...")
print(f"Testing video: {video_path}")

try:
    vr = decord.VideoReader(video_path)
    frames = len(vr)
    print(f"✅ Successfully loaded video: {frames} frames")
    
    # Try to get some frames to trigger more decoding
    if frames > 0:
        sample_indices = list(range(min(5, frames)))
        frame_data = vr.get_batch(sample_indices)
        print(f"✅ Successfully decoded {len(sample_indices)} frames")
        
except Exception as e:
    print(f"❌ Error: {e}")

print("\nTest completed. If HEVC SPS messages appeared above, the suppression didn't work.")
print("If no HEVC messages appeared, the suppression is working!")