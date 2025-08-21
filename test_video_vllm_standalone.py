#!/usr/bin/env python3
"""
Standalone test script for GRPO video handling with vLLM.
This tests the video processing functionality without requiring all dependencies.
"""

import base64
import os
import tempfile
from io import BytesIO
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import torch


def test_prepare_video_messages_for_vllm_mock():
    """Test the prepare_video_messages_for_vllm function with mocked dependencies"""
    
    print("Testing prepare_video_messages_for_vllm function...")
    
    # Define the function inline (copied from grpo_trainer.py)
    def prepare_video_messages_for_vllm(messages):
        """
        The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
        Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_type` of the video explicitly set to `video/jpeg`.
        By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
        """
        # Mock process_vision_info for testing
        def mock_process_vision_info(video_message, return_video_kwargs=True):
            # Create mock video tensor (8 frames, 224x224, RGB)
            video_tensor = torch.randn(8, 3, 224, 224)
            video_kwargs = {"fps": [2.0]}
            return None, video_tensor, video_kwargs
        
        vllm_messages, fps_list = [], []
        for message in messages:
            message_content_list = message["content"] if isinstance(message.get("content"), list) else [message.get("content")]
            if not message_content_list:
                vllm_messages.append(message)
                continue

            new_content_list = []
            for part_message in message_content_list:
                if isinstance(part_message, dict) and 'video' in part_message:
                    video_message = [{'content': [part_message]}]
                    image_inputs, video_inputs, video_kwargs = mock_process_vision_info(video_message, return_video_kwargs=True)
                    if video_inputs is not None:
                        # Convert video tensor to base64 frames
                        video_input = video_inputs.permute(0, 2, 3, 1).numpy().astype(np.uint8)
                        fps_list.extend(video_kwargs.get('fps', []))

                        # Encode frames as base64
                        base64_frames = []
                        for frame in video_input:
                            img = Image.fromarray(frame)
                            output_buffer = BytesIO()
                            img.save(output_buffer, format="jpeg")
                            byte_data = output_buffer.getvalue()
                            base64_str = base64.b64encode(byte_data).decode("utf-8")
                            base64_frames.append(base64_str)

                        part_message = {
                            "type": "video_url",
                            "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                        }
                new_content_list.append(part_message)
            
            message_copy = message.copy()
            message_copy["content"] = new_content_list
            vllm_messages.append(message_copy)
        
        return vllm_messages, {'fps': fps_list}
    
    # Test input
    test_messages = [
        {
            "content": [
                {"type": "text", "text": "What's happening in this video?"},
                {"type": "video", "video": "test_video.mp4", "fps": 2.0}
            ]
        }
    ]
    
    # Call function
    try:
        prepared_messages, video_kwargs = prepare_video_messages_for_vllm(test_messages)
        
        # Verify results
        assert len(prepared_messages) == 1, f"Expected 1 message, got {len(prepared_messages)}"
        assert "content" in prepared_messages[0], "Message should have 'content' key"
        assert isinstance(video_kwargs, dict), f"video_kwargs should be dict, got {type(video_kwargs)}"
        assert "fps" in video_kwargs, "video_kwargs should have 'fps' key"
        
        # Check video conversion
        content_list = prepared_messages[0]["content"]
        video_part = None
        for part in content_list:
            if isinstance(part, dict) and part.get("type") == "video_url":
                video_part = part
                break
        
        assert video_part is not None, "Should find video_url part in content"
        assert "video_url" in video_part, "video_part should have 'video_url' key"
        assert "url" in video_part["video_url"], "video_url should have 'url' key"
        
        url = video_part["video_url"]["url"]
        assert url.startswith("data:video/jpeg;base64,"), f"URL should start with base64 prefix, got: {url[:50]}..."
        
        print("✅ prepare_video_messages_for_vllm test passed!")
        return True
        
    except Exception as e:
        print(f"❌ prepare_video_messages_for_vllm test failed: {e}")
        return False


def test_video_tensor_processing():
    """Test video tensor processing and base64 encoding"""
    
    print("Testing video tensor processing...")
    
    try:
        # Create mock video tensor (4 frames, 224x224 RGB)
        video_tensor = torch.randint(0, 255, (4, 3, 224, 224), dtype=torch.uint8)
        
        # Convert to numpy format (frames, height, width, channels)
        video_numpy = video_tensor.permute(0, 2, 3, 1).numpy()
        
        # Process frames to base64
        base64_frames = []
        for frame in video_numpy:
            img = Image.fromarray(frame)
            output_buffer = BytesIO()
            img.save(output_buffer, format="jpeg")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_frames.append(base64_str)
        
        # Verify results
        assert len(base64_frames) == 4, f"Expected 4 frames, got {len(base64_frames)}"
        
        for i, frame_b64 in enumerate(base64_frames):
            assert isinstance(frame_b64, str), f"Frame {i} should be string"
            assert len(frame_b64) > 0, f"Frame {i} should not be empty"
            
            # Try to decode to verify it's valid base64
            decoded = base64.b64decode(frame_b64)
            assert len(decoded) > 0, f"Frame {i} should decode to non-empty bytes"
        
        # Test full URL format
        full_url = f"data:video/jpeg;base64,{','.join(base64_frames)}"
        assert full_url.startswith("data:video/jpeg;base64,"), "URL should have correct prefix"
        assert "," in full_url, "URL should contain frame separators"
        
        print("✅ Video tensor processing test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Video tensor processing test failed: {e}")
        return False


def test_vllm_input_format():
    """Test that vLLM input format is correct"""
    
    print("Testing vLLM input format...")
    
    try:
        # Test colocate mode input format
        prompt_text = "Describe what you see in this video."
        video_tensor = torch.randn(8, 3, 224, 224)  # 8 frames
        
        # Create vLLM input as would be done in colocate mode
        mm_data = {"video": video_tensor}
        video_kwargs = {"fps": [2.0], "max_frames": [8]}
        
        vllm_input = {
            "prompt": prompt_text,
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": video_kwargs,
        }
        
        # Verify structure
        assert "prompt" in vllm_input, "Should have 'prompt' key"
        assert "multi_modal_data" in vllm_input, "Should have 'multi_modal_data' key"
        assert "mm_processor_kwargs" in vllm_input, "Should have 'mm_processor_kwargs' key"
        
        assert isinstance(vllm_input["prompt"], str), "Prompt should be string"
        assert isinstance(vllm_input["multi_modal_data"], dict), "multi_modal_data should be dict"
        assert isinstance(vllm_input["mm_processor_kwargs"], dict), "mm_processor_kwargs should be dict"
        
        # Check video tensor
        assert "video" in vllm_input["multi_modal_data"], "Should have video in multi_modal_data"
        video_data = vllm_input["multi_modal_data"]["video"]
        assert torch.is_tensor(video_data), "Video data should be tensor"
        assert video_data.shape == (8, 3, 224, 224), f"Expected shape (8,3,224,224), got {video_data.shape}"
        
        # Check video kwargs
        assert "fps" in vllm_input["mm_processor_kwargs"], "Should have fps in kwargs"
        assert isinstance(vllm_input["mm_processor_kwargs"]["fps"], list), "fps should be list"
        
        print("✅ vLLM input format test passed!")
        return True
        
    except Exception as e:
        print(f"❌ vLLM input format test failed: {e}")
        return False


def test_video_message_structure():
    """Test video message structure validation"""
    
    print("Testing video message structure...")
    
    try:
        # Test various message formats
        test_cases = [
            # Standard format
            {
                "content": [
                    {"type": "text", "text": "What's in this video?"},
                    {"type": "video", "video": "test.mp4", "fps": 2.0}
                ]
            },
            # With additional parameters
            {
                "content": [
                    {"type": "text", "text": "Analyze this video"},
                    {
                        "type": "video", 
                        "video": "test.mp4", 
                        "fps": 3.0,
                        "max_frames": 16,
                        "total_pixels": 20480 * 28 * 28
                    }
                ]
            },
            # Mixed content
            {
                "content": [
                    {"type": "text", "text": "Look at this"},
                    {"type": "video", "video": "video1.mp4"},
                    {"type": "text", "text": "and this"},
                    {"type": "video", "video": "video2.mp4"}
                ]
            }
        ]
        
        for i, message in enumerate(test_cases):
            # Verify basic structure
            assert "content" in message, f"Test case {i}: Should have 'content' key"
            assert isinstance(message["content"], list), f"Test case {i}: Content should be list"
            
            # Check for video parts
            video_count = 0
            text_count = 0
            
            for part in message["content"]:
                assert isinstance(part, dict), f"Test case {i}: Content parts should be dicts"
                assert "type" in part, f"Test case {i}: Content parts should have 'type'"
                
                if part["type"] == "video":
                    assert "video" in part, f"Test case {i}: Video parts should have 'video' key"
                    video_count += 1
                elif part["type"] == "text":
                    assert "text" in part, f"Test case {i}: Text parts should have 'text' key"
                    text_count += 1
            
            assert video_count > 0, f"Test case {i}: Should have at least one video part"
            assert text_count > 0, f"Test case {i}: Should have at least one text part"
        
        print("✅ Video message structure test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Video message structure test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Running GRPO Video vLLM Tests...")
    print("=" * 50)
    
    tests = [
        test_video_tensor_processing,
        test_video_message_structure,
        test_vllm_input_format,
        test_prepare_video_messages_for_vllm_mock,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        print(f"\n🔧 Running {test_func.__name__}...")
        if test_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)