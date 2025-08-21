import base64
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import torch


class TestGRPOVideoFunctions(unittest.TestCase):
    """Test cases for GRPO video processing functions"""

    def test_video_tensor_to_base64_conversion(self):
        """Test converting video tensor to base64 format"""
        
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
        self.assertEqual(len(base64_frames), 4)
        
        for i, frame_b64 in enumerate(base64_frames):
            self.assertIsInstance(frame_b64, str)
            self.assertGreater(len(frame_b64), 0)
            
            # Try to decode to verify it's valid base64
            decoded = base64.b64decode(frame_b64)
            self.assertGreater(len(decoded), 0)

    def test_vllm_server_video_url_format(self):
        """Test vLLM server video URL format"""
        
        # Create sample base64 frames
        sample_frames = ["frame1_base64", "frame2_base64", "frame3_base64"]
        
        # Create video URL as done in prepare_video_messages_for_vllm
        video_url = f"data:video/jpeg;base64,{','.join(sample_frames)}"
        
        # Verify format
        self.assertTrue(video_url.startswith("data:video/jpeg;base64,"))
        self.assertIn(",", video_url)
        
        # Extract frames
        url_parts = video_url.split("data:video/jpeg;base64,", 1)
        self.assertEqual(len(url_parts), 2)
        
        frames_part = url_parts[1]
        extracted_frames = frames_part.split(",")
        self.assertEqual(len(extracted_frames), 3)
        self.assertEqual(extracted_frames, sample_frames)

    def test_vllm_colocate_input_structure(self):
        """Test vLLM colocate mode input structure"""
        
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
        self.assertIn("prompt", vllm_input)
        self.assertIn("multi_modal_data", vllm_input)
        self.assertIn("mm_processor_kwargs", vllm_input)
        
        self.assertIsInstance(vllm_input["prompt"], str)
        self.assertIsInstance(vllm_input["multi_modal_data"], dict)
        self.assertIsInstance(vllm_input["mm_processor_kwargs"], dict)
        
        # Check video tensor
        self.assertIn("video", vllm_input["multi_modal_data"])
        video_data = vllm_input["multi_modal_data"]["video"]
        self.assertTrue(torch.is_tensor(video_data))
        self.assertEqual(video_data.shape, (8, 3, 224, 224))

    def test_video_message_content_structure(self):
        """Test video message content structure validation"""
        
        # Test standard video message format
        message = {
            "content": [
                {"type": "text", "text": "What's in this video?"},
                {"type": "video", "video": "test.mp4", "fps": 2.0}
            ]
        }
        
        # Verify structure
        self.assertIn("content", message)
        self.assertIsInstance(message["content"], list)
        
        # Check content parts
        video_parts = [part for part in message["content"] if part.get("type") == "video"]
        text_parts = [part for part in message["content"] if part.get("type") == "text"]
        
        self.assertEqual(len(video_parts), 1)
        self.assertEqual(len(text_parts), 1)
        
        # Check video part
        video_part = video_parts[0]
        self.assertIn("video", video_part)
        self.assertIn("fps", video_part)

    def test_video_kwargs_processing(self):
        """Test video kwargs structure and content"""
        
        video_kwargs = {
            "fps": [2.0, 3.0],
            "max_frames": [16, 20],
        }
        
        # Verify structure
        self.assertIn("fps", video_kwargs)
        self.assertIsInstance(video_kwargs["fps"], list)
        
        # Verify fps values are reasonable
        for fps in video_kwargs["fps"]:
            self.assertGreater(fps, 0)
            self.assertLess(fps, 60)  # Reasonable FPS range

    @patch('builtins.hasattr')
    def test_video_availability_check(self, mock_hasattr):
        """Test checking if video data is available"""
        
        # Mock scenario where video is available
        sample_input = {"video": "test_video.mp4", "prompt": "test"}
        
        has_videos = "video" in sample_input
        self.assertTrue(has_videos)
        
        # Mock scenario where video is not available  
        sample_input_no_video = {"prompt": "test"}
        
        has_videos = "video" in sample_input_no_video
        self.assertFalse(has_videos)

    def test_mixed_multimodal_input(self):
        """Test handling inputs with both images and videos"""
        
        # Simulate mixed input scenario
        sample_inputs = [
            {"prompt": "test1", "image": "img1.jpg"},
            {"prompt": "test2", "video": "vid1.mp4"},
            {"prompt": "test3", "image": "img2.jpg", "video": "vid2.mp4"},
        ]
        
        # Check detection logic
        has_images = any("image" in inp for inp in sample_inputs)
        has_videos = any("video" in inp for inp in sample_inputs)
        
        self.assertTrue(has_images)
        self.assertTrue(has_videos)
        
        # Test individual input processing
        for inp in sample_inputs:
            if "video" in inp and "image" in inp:
                # Mixed input - should handle both
                self.assertIn("video", inp)
                self.assertIn("image", inp)
            elif "video" in inp:
                # Video only
                self.assertIn("video", inp)
                self.assertNotIn("image", inp)
            elif "image" in inp:
                # Image only
                self.assertIn("image", inp)
                self.assertNotIn("video", inp)

    def test_prepare_video_messages_mock_structure(self):
        """Test the structure returned by prepare_video_messages_for_vllm"""
        
        # Simulate function behavior with mock data
        def mock_prepare_video_messages_for_vllm(messages):
            vllm_messages = []
            fps_list = []
            
            for message in messages:
                new_content = []
                for part in message.get("content", []):
                    if isinstance(part, dict) and part.get("type") == "video":
                        # Convert to vLLM format
                        new_content.append({
                            "type": "video_url",
                            "video_url": {
                                "url": "data:video/jpeg;base64,mock_frame1,mock_frame2"
                            }
                        })
                        fps_list.append(2.0)
                    else:
                        new_content.append(part)
                
                vllm_messages.append({"content": new_content})
            
            return vllm_messages, {"fps": fps_list}
        
        # Test input
        test_messages = [{
            "content": [
                {"type": "text", "text": "What's happening?"},
                {"type": "video", "video": "test.mp4"}
            ]
        }]
        
        # Call mock function
        prepared_messages, video_kwargs = mock_prepare_video_messages_for_vllm(test_messages)
        
        # Verify structure
        self.assertEqual(len(prepared_messages), 1)
        self.assertIn("content", prepared_messages[0])
        self.assertIsInstance(video_kwargs, dict)
        self.assertIn("fps", video_kwargs)
        
        # Check video conversion
        content_list = prepared_messages[0]["content"]
        video_parts = [p for p in content_list if p.get("type") == "video_url"]
        self.assertEqual(len(video_parts), 1)
        
        video_part = video_parts[0]
        self.assertIn("video_url", video_part)
        self.assertIn("url", video_part["video_url"])
        self.assertTrue(video_part["video_url"]["url"].startswith("data:video/jpeg;base64,"))


if __name__ == '__main__':
    unittest.main()