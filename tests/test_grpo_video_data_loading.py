import json
import os
import tempfile
import unittest
import glob
from pathlib import Path

import pytest
from datasets import Dataset


def load_multiple_datasets(data_dir="./data", max_samples_per_dataset=3000):
    """Load multiple video datasets from data directory"""
    all_data = []
    
    # Automatically discover all JSON files in the data directory
    json_pattern = os.path.join(data_dir, "*.json")
    json_file_paths = glob.glob(json_pattern)
    
    # Extract just the filenames from full paths
    json_files = [os.path.basename(path) for path in json_file_paths]
    
    # Sort for consistent order
    json_files.sort()
    
    print(f"Found {len(json_files)} JSON files in {os.path.abspath(data_dir)}:")
    for json_file in json_files:
        print(f"  - {json_file}")
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        print(f"file_path: {os.path.abspath(file_path)}")
        if not os.path.exists(file_path):
            print(f"Skipping {json_file}: file not found")
            continue
            
        print(f"Loading {json_file}...")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {json_file}")
            continue
        
        dataset_data = []
        for i, item in enumerate(annotations[:max_samples_per_dataset]):
            try:
                # Handle different data structures
                conversations = item.get('conversations', [])
                videos = item.get('videos', [])
                
                if not conversations or not videos:
                    continue
                    
                # Extract messages with compatibility for different formats
                system_msg = "You are a helpful assistant."
                human_msg = None
                assistant_msg = None
                
                for conv in conversations:
                    role = conv.get('from', conv.get('role', ''))
                    content = conv.get('value', conv.get('content', ''))
                    
                    if role in ['human', 'user']:
                        human_msg = content
                    elif role in ['gpt', 'assistant']:
                        assistant_msg = content
                
                if human_msg and assistant_msg:
                    dataset_data.append({
                        "video": videos[0],  # now only support 1 video
                        "prompt": [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": human_msg}
                        ],
                        "solution": assistant_msg,
                        "dataset": json_file.split('_')[0]  # Add dataset identifier
                    })
                    
            except Exception as e:
                continue
        
        print(f"  Loaded {len(dataset_data)} samples from {json_file}")
        all_data.extend(dataset_data)
    
    print(f"Total samples loaded: {len(all_data)}")
    return Dataset.from_list(all_data)


class TestGRPOVideoDataLoading(unittest.TestCase):
    """Test cases for GRPO video data loading functionality"""

    def setUp(self):
        """Set up test fixtures with temporary directories and mock data"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock dataset with different conversation formats
        self.mock_data_format1 = [
            {
                "conversations": [
                    {"from": "human", "value": "What is happening in this video?"},
                    {"from": "gpt", "value": '{"description": "test description", "actions": ["walking", "running"], "objects": ["person", "dog"], "characteristics": ["outdoor", "sunny"], "title": "Test Video"}'}
                ],
                "videos": ["video1.mp4"]
            },
            {
                "conversations": [
                    {"from": "user", "value": "Describe the scene"},
                    {"from": "assistant", "value": '{"description": "another test", "actions": ["sitting"], "objects": ["chair"], "characteristics": ["indoor"], "title": "Indoor Scene"}'}
                ],
                "videos": ["video2.mp4"]
            }
        ]
        
        self.mock_data_format2 = [
            {
                "conversations": [
                    {"role": "user", "content": "What do you see?"},
                    {"role": "assistant", "content": '{"description": "format2 test", "actions": ["dancing"], "objects": ["stage"], "characteristics": ["night"], "title": "Dance Performance"}'}
                ],
                "videos": ["video3.mp4"]
            }
        ]
        
        # Create test JSON files
        self.test_file1 = os.path.join(self.temp_dir, "kids_videos.json")
        self.test_file2 = os.path.join(self.temp_dir, "adults_videos.json")
        
        with open(self.test_file1, 'w') as f:
            json.dump(self.mock_data_format1, f)
        
        with open(self.test_file2, 'w') as f:
            json.dump(self.mock_data_format2, f)

    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_single_dataset_file(self):
        """Test loading from a single JSON file"""
        # Create a directory with only one file
        single_file_dir = tempfile.mkdtemp()
        single_file = os.path.join(single_file_dir, "single_test.json")
        
        with open(single_file, 'w') as f:
            json.dump(self.mock_data_format1, f)
        
        try:
            dataset = load_multiple_datasets(single_file_dir, max_samples_per_dataset=10)
            
            self.assertIsInstance(dataset, Dataset)
            self.assertEqual(len(dataset), 2)
            
            # Check required fields are present
            for item in dataset:
                self.assertIn("video", item)
                self.assertIn("prompt", item)
                self.assertIn("solution", item)
                self.assertIn("dataset", item)
                
            # Check prompt structure
            first_item = dataset[0]
            self.assertEqual(len(first_item["prompt"]), 2)
            self.assertEqual(first_item["prompt"][0]["role"], "system")
            self.assertEqual(first_item["prompt"][1]["role"], "user")
            
        finally:
            import shutil
            shutil.rmtree(single_file_dir)

    def test_load_multiple_datasets(self):
        """Test loading from multiple JSON files"""
        dataset = load_multiple_datasets(self.temp_dir, max_samples_per_dataset=10)
        
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(len(dataset), 3)  # 2 from first file + 1 from second file
        
        # Check that data from both files is included
        dataset_identifiers = [item["dataset"] for item in dataset]
        self.assertIn("kids", dataset_identifiers)
        self.assertIn("adults", dataset_identifiers)

    def test_max_samples_per_dataset_limit(self):
        """Test that max_samples_per_dataset parameter works correctly"""
        dataset = load_multiple_datasets(self.temp_dir, max_samples_per_dataset=1)
        
        # Should only load 1 sample per file (2 files = 2 samples max)
        self.assertLessEqual(len(dataset), 2)

    def test_empty_directory(self):
        """Test behavior with empty directory"""
        empty_dir = tempfile.mkdtemp()
        try:
            dataset = load_multiple_datasets(empty_dir)
            self.assertEqual(len(dataset), 0)
        finally:
            import shutil
            shutil.rmtree(empty_dir)

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON files"""
        invalid_dir = tempfile.mkdtemp()
        invalid_file = os.path.join(invalid_dir, "invalid.json")
        
        # Create invalid JSON
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        try:
            # Should handle gracefully without crashing
            dataset = load_multiple_datasets(invalid_dir)
            self.assertEqual(len(dataset), 0)
        finally:
            import shutil
            shutil.rmtree(invalid_dir)

    def test_missing_required_fields(self):
        """Test handling of data with missing required fields"""
        incomplete_data = [
            {"conversations": []},  # Missing videos
            {"videos": ["test.mp4"]},  # Missing conversations
            {"conversations": [{"from": "human", "value": "test"}], "videos": ["test.mp4"]}  # Missing assistant response
        ]
        
        incomplete_dir = tempfile.mkdtemp()
        incomplete_file = os.path.join(incomplete_dir, "incomplete.json")
        
        with open(incomplete_file, 'w') as f:
            json.dump(incomplete_data, f)
        
        try:
            dataset = load_multiple_datasets(incomplete_dir)
            # Should filter out invalid entries
            self.assertEqual(len(dataset), 0)
        finally:
            import shutil
            shutil.rmtree(incomplete_dir)

    def test_train_test_split(self):
        """Test train/test dataset splitting functionality"""
        dataset = load_multiple_datasets(self.temp_dir, max_samples_per_dataset=10)
        
        # Perform train/test split as done in the main script
        split_dataset = dataset.train_test_split(test_size=2, seed=42)
        
        self.assertIn("train", split_dataset)
        self.assertIn("test", split_dataset)
        
        train_dataset = split_dataset["train"]
        eval_dataset = split_dataset["test"]
        
        # Check that split maintains data structure
        self.assertIsInstance(train_dataset, Dataset)
        self.assertIsInstance(eval_dataset, Dataset)
        
        # Check split sizes
        self.assertEqual(len(eval_dataset), 2)
        self.assertEqual(len(train_dataset), len(dataset) - 2)
        
        # Verify no data loss
        self.assertEqual(len(train_dataset) + len(eval_dataset), len(dataset))

    def test_dataset_structure_consistency(self):
        """Test that loaded dataset has consistent structure across all samples"""
        dataset = load_multiple_datasets(self.temp_dir, max_samples_per_dataset=10)
        
        required_keys = {"video", "prompt", "solution", "dataset"}
        
        for i, sample in enumerate(dataset):
            # Check all required keys are present
            self.assertEqual(set(sample.keys()), required_keys, 
                           f"Sample {i} missing required keys")
            
            # Check prompt structure
            self.assertIsInstance(sample["prompt"], list)
            self.assertEqual(len(sample["prompt"]), 2)
            self.assertEqual(sample["prompt"][0]["role"], "system")
            self.assertEqual(sample["prompt"][1]["role"], "user")
            
            # Check types
            self.assertIsInstance(sample["video"], str)
            self.assertIsInstance(sample["solution"], str)
            self.assertIsInstance(sample["dataset"], str)

    def test_different_conversation_formats(self):
        """Test that different conversation formats are handled correctly"""
        dataset = load_multiple_datasets(self.temp_dir, max_samples_per_dataset=10)
        
        # Should handle both 'from'/'value' and 'role'/'content' formats
        self.assertGreater(len(dataset), 0)
        
        # Verify data from both formats is present
        solutions = [item["solution"] for item in dataset]
        self.assertTrue(any("format2 test" in sol for sol in solutions))
        self.assertTrue(any("test description" in sol for sol in solutions))

    def test_dataset_identifier_extraction(self):
        """Test that dataset identifiers are correctly extracted from filenames"""
        dataset = load_multiple_datasets(self.temp_dir, max_samples_per_dataset=10)
        
        dataset_ids = set(item["dataset"] for item in dataset)
        expected_ids = {"kids", "adults"}
        
        self.assertEqual(dataset_ids, expected_ids)


if __name__ == '__main__':
    unittest.main()