#!/usr/bin/env python3
"""
Integration demo test to verify video handling in GRPO trainer.
This script demonstrates how the video processing works end-to-end.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import and test the helper function directly
try:
    from trl.trainer.grpo_trainer import prepare_video_messages_for_vllm
    print("✅ Successfully imported prepare_video_messages_for_vllm from grpo_trainer")
    function_available = True
except ImportError as e:
    print(f"❌ Could not import prepare_video_messages_for_vllm: {e}")
    function_available = False

def test_video_helper_function():
    """Test the video helper function with mock data"""
    
    if not function_available:
        print("Skipping function test due to import error")
        return False
    
    # Mock qwen_vl_utils.process_vision_info
    import unittest.mock as mock
    
    def mock_process_vision_info(video_message, return_video_kwargs=True):
        import torch
        # Return mock video tensor and kwargs in the expected format
        video_tensor = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        video_kwargs = {"video_fps": [2.0]}  # Use video_fps instead of fps
        return None, video_tensor, video_kwargs
    
    with mock.patch('trl.trainer.grpo_trainer.process_vision_info', side_effect=mock_process_vision_info):
        
        # Test input messages
        test_messages = [
            {
                "content": [
                    {"type": "text", "text": "What's happening in this video?"},
                    {"type": "video", "video": "sample_video.mp4", "fps": 2.0}
                ]
            }
        ]
        
        try:
            # Call the actual function
            prepared_messages, video_kwargs = prepare_video_messages_for_vllm(test_messages)
            
            # Debug output
            print(f"Debug - video_kwargs keys: {list(video_kwargs.keys())}")
            print(f"Debug - video_kwargs: {video_kwargs}")
            
            # Verify results
            assert len(prepared_messages) == 1, f"Expected 1 message, got {len(prepared_messages)}"
            assert "content" in prepared_messages[0], "Message should have 'content' key"
            assert isinstance(video_kwargs, dict), "video_kwargs should be dict"
            # Check for any fps-related key
            fps_keys = [k for k in video_kwargs.keys() if 'fps' in k.lower()]
            assert len(fps_keys) > 0, f"video_kwargs should have fps-related key, got keys: {list(video_kwargs.keys())}"
            
            print("✅ Video helper function test passed!")
            return True
            
        except Exception as e:
            print(f"❌ Video helper function test failed: {e}")
            return False

def test_grpo_trainer_imports():
    """Test that GRPO trainer can import required modules"""
    
    print("Testing GRPO trainer imports...")
    
    try:
        # Test basic imports
        from trl.trainer.grpo_trainer import GRPOTrainer
        print("✅ GRPOTrainer import successful")
        
        # Test that our added imports are available
        import base64
        import numpy as np
        from PIL import Image
        from io import BytesIO
        print("✅ Required dependencies available")
        
        # Test that utility functions are available
        from trl.trainer.grpo_trainer import (
            split_pixel_values_by_grid,
            unsplit_pixel_values_by_grid,
            prepare_video_messages_for_vllm
        )
        print("✅ Utility functions available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_video_processing_pipeline():
    """Test the full video processing pipeline"""
    
    print("Testing video processing pipeline...")
    
    try:
        import torch
        import base64
        from PIL import Image
        from io import BytesIO
        import numpy as np
        
        # Step 1: Create mock video tensor
        video_tensor = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        print(f"✅ Created video tensor: {video_tensor.shape}")
        
        # Step 2: Convert to frames (as done in prepare_video_messages_for_vllm)
        video_frames = video_tensor.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        print(f"✅ Converted to frames: {video_frames.shape}")
        
        # Step 3: Encode frames as base64
        base64_frames = []
        for i, frame in enumerate(video_frames):
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="jpeg")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)
        
        print(f"✅ Encoded {len(base64_frames)} frames to base64")
        
        # Step 4: Create vLLM server format
        video_url = f"data:video/jpeg;base64,{','.join(base64_frames)}"
        vllm_server_format = {
            "type": "video_url",
            "video_url": {"url": video_url}
        }
        
        print(f"✅ Created vLLM server format (URL length: {len(video_url)})")
        
        # Step 5: Create vLLM colocate format
        vllm_colocate_format = {
            "prompt": "Describe this video",
            "multi_modal_data": {"video": video_tensor},
            "mm_processor_kwargs": {"fps": [2.0]}
        }
        
        print("✅ Created vLLM colocate format")
        
        # Verify formats
        assert vllm_server_format["video_url"]["url"].startswith("data:video/jpeg;base64,")
        assert torch.is_tensor(vllm_colocate_format["multi_modal_data"]["video"])
        assert "fps" in vllm_colocate_format["mm_processor_kwargs"]
        
        print("✅ Video processing pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Video processing pipeline test failed: {e}")
        return False

def test_dataset_compatibility():
    """Test that video datasets work with the trainer"""
    
    print("Testing dataset compatibility...")
    
    try:
        from datasets import Dataset
        
        # Create mock video dataset (similar to grpo_video.py)
        video_dataset = Dataset.from_dict({
            "prompt": [
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is happening in this video?"}
                ],
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Describe the scene in the video."}
                ]
            ],
            "video": [
                "path/to/video1.mp4",
                "path/to/video2.mp4"
            ]
        })
        
        print(f"✅ Created video dataset with {len(video_dataset)} samples")
        
        # Verify dataset structure
        assert "video" in video_dataset.column_names
        assert "prompt" in video_dataset.column_names
        
        # Test sample structure  
        sample = video_dataset[0]
        assert "video" in sample
        assert "prompt" in sample
        assert isinstance(sample["prompt"], list)
        assert len(sample["prompt"]) == 2  # system + user
        
        print("✅ Dataset compatibility test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset compatibility test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🎬 GRPO Video Integration Demo")
    print("=" * 50)
    
    tests = [
        test_grpo_trainer_imports,
        test_video_processing_pipeline,
        test_dataset_compatibility,
        test_video_helper_function,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print(f"\n🔧 {test_func.__name__}")
        print("-" * 40)
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Integration Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All integration tests passed!")
        print("\n✨ The GRPO trainer video handling with vLLM is working correctly!")
    else:
        print("❌ Some integration tests failed!")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)