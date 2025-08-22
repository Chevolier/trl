#!/usr/bin/env python3

import json
import os
from tqdm import tqdm

def quick_filter_test(data_dir="/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video", sample_size=50):
    """Quick test of filtering with small sample from each dataset"""
    
    json_files = [
        "0_30_eufy_videos.json",
        "12_actionData_videos.json", 
        "kids_videos.json",
        "smarthome_videos_2.json",
        "videos_val_oops.json"
    ]
    
    total_results = {}
    
    print(f"🧪 Quick filter test with {sample_size} samples per dataset")
    print("=" * 60)
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        if not os.path.exists(file_path):
            print(f"❌ {json_file}: file not found")
            continue
            
        print(f"\n📂 {json_file}")
        
        # Load data
        with open(file_path, 'r') as f:
            annotations = json.load(f)
        
        # Take sample
        sample_data = annotations[:sample_size]
        original_count = len(sample_data)
        print(f"   Testing {original_count} items...")
        
        valid_items = []
        corrupt_items = []
        
        for i, item in enumerate(tqdm(sample_data, desc="   Processing", leave=False)):
            try:
                videos = item.get('videos', [])
                if not videos:
                    corrupt_items.append(item)
                    continue
                
                all_valid = True
                issues = []
                
                for video in videos:
                    video_path = os.path.join(data_dir, video)
                    
                    if not os.path.exists(video_path):
                        all_valid = False
                        issues.append(f"not_found: {video}")
                        continue
                    
                    file_size = os.path.getsize(video_path)
                    if file_size <= 1024:
                        all_valid = False
                        issues.append(f"too_small: {video}")
                        continue
                    
                    try:
                        import decord
                        # Suppress stderr during decord reading
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
                            all_valid = False
                            issues.append(f"no_frames: {video}")
                    except Exception as e:
                        all_valid = False
                        issues.append(f"decord_error: {video}")
                
                if all_valid:
                    valid_items.append(item)
                else:
                    corrupt_item = item.copy()
                    corrupt_item['_corruption_reason'] = issues
                    corrupt_items.append(corrupt_item)
                    
            except Exception as e:
                corrupt_item = item.copy()
                corrupt_item['_corruption_reason'] = [f"processing_error: {e}"]
                corrupt_items.append(corrupt_item)
        
        # Results
        valid_count = len(valid_items)
        corrupt_count = len(corrupt_items)
        success_rate = valid_count / original_count * 100
        
        print(f"   ✅ Valid: {valid_count}")
        print(f"   ❌ Corrupt: {corrupt_count}")  
        print(f"   📊 Success rate: {success_rate:.1f}%")
        
        # Store results
        total_results[json_file] = {
            'sample_size': original_count,
            'valid': valid_count,
            'corrupt': corrupt_count,
            'success_rate': success_rate
        }
        
        # Show some corruption reasons
        if corrupt_items:
            corruption_types = {}
            for item in corrupt_items:
                reasons = item.get('_corruption_reason', [])
                for reason in reasons:
                    reason_type = reason.split(':')[0]
                    corruption_types[reason_type] = corruption_types.get(reason_type, 0) + 1
            
            print(f"   🔍 Corruption breakdown:")
            for reason_type, count in corruption_types.items():
                print(f"      {reason_type}: {count}")
    
    # Overall summary
    print(f"\n" + "=" * 60)
    print(f"📊 OVERALL SUMMARY")
    print(f"=" * 60)
    
    total_samples = sum(r['sample_size'] for r in total_results.values())
    total_valid = sum(r['valid'] for r in total_results.values()) 
    total_corrupt = sum(r['corrupt'] for r in total_results.values())
    overall_rate = total_valid / total_samples * 100 if total_samples > 0 else 0
    
    print(f"Total samples tested: {total_samples}")
    print(f"Total valid: {total_valid}")
    print(f"Total corrupt: {total_corrupt}")
    print(f"Overall success rate: {overall_rate:.1f}%")
    
    print(f"\nPer-dataset breakdown:")
    for filename, results in total_results.items():
        print(f"  {filename}: {results['success_rate']:.1f}% ({results['valid']}/{results['sample_size']})")
    
    return total_results

if __name__ == "__main__":
    quick_filter_test()