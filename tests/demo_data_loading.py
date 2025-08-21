#!/usr/bin/env python3
"""
Demo script to load kids_videos.json data using load_multiple_datasets() 
and split between train and test, showing dataset types and examples.
"""

import json
import os
import glob
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
        # Handle case where annotations might not be a list
        if isinstance(annotations, list):
            items_to_process = annotations[:max_samples_per_dataset]
        else:
            print(f"  Skipping {json_file}: not a list format")
            continue
            
        for i, item in enumerate(items_to_process):
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


def main():
    # Load data from the specified path
    data_path = "/home/ec2-user/SageMaker/efs/Projects/vlm-rl-training/data/video"
    
    print("=" * 60)
    print("LOADING DATA WITH load_multiple_datasets()")
    print("=" * 60)
    
    # Load dataset
    dataset = load_multiple_datasets(data_path, max_samples_per_dataset=3000)
    print(f"\nTotal dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No data found. Exiting.")
        return
    
    print("\n" + "=" * 60)
    print("SPLITTING INTO TRAIN/TEST")
    print("=" * 60)
    
    # Split into train/test as done in grpo_video.py
    split_dataset = dataset.train_test_split(test_size=100, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    
    # Show dataset information
    print(f"\nTrain dataset type: {type(train_dataset)}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    print("\n" + "=" * 60)
    print("TRAIN DATASET EXAMPLE")
    print("=" * 60)
    
    if len(train_dataset) > 0:
        train_example = dict(train_dataset[0])
        for key, value in train_example.items():
            if isinstance(value, str) and len(value) > 150:
                print(f"{key}: {value[:150]}...")
            else:
                print(f"{key}: {value}")
    else:
        print("No train examples available")
    
    print("\n" + "=" * 60)
    print("TEST DATASET EXAMPLE") 
    print("=" * 60)
    
    if len(test_dataset) > 0:
        test_example = dict(test_dataset[0])
        for key, value in test_example.items():
            if isinstance(value, str) and len(value) > 150:
                print(f"{key}: {value[:150]}...")
            else:
                print(f"{key}: {value}")
    else:
        print("No test examples available")


if __name__ == "__main__":
    main()