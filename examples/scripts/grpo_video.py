#!/usr/bin/env python3

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
# Suppress vLLM INFO messages
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.engine").setLevel(logging.WARNING)
logging.getLogger("vllm.worker").setLevel(logging.WARNING)

import os
# os.environ["HF_HUB_OFFLINE"] = "1"  # 避免HuggingFace API限流
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
# Suppress HEVC SPS error messages while keeping other errors visible
import json
import subprocess
import torch
from PIL import Image
from datasets import Dataset
from pathlib import Path

# Import decord early to set log level
import decord
# Suppress HEVC SPS warning messages (set to QUIET level)
decord.logging.set_level(decord.logging.QUIET)

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser
)

from sentence_transformers import SentenceTransformer, util
import re
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def extract_json_from_string(json_string):
    """从可能包含 Markdown 代码块的字符串中提取并解析 JSON 对象"""
    pattern = r".*?```json\s*([\s\S]*?)```.*?"
    match = re.search(pattern, json_string)
    
    if match:
        return match.group(1)
    else:
        return json_string


sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', local_files_only=False)
sbert_model = sbert_model.to('cpu')

sent_embedding = SentenceTransformer('/home/ec2-user/SageMaker/efs/Models/Qwen3-Embedding-0.6B', local_files_only=False)

def summary_reward_bert(completions, **kwargs):
    # print(f"REWARD FUNCTION CALLED with {len(completions)} completions")
    # print(f"Available kwargs keys: {list(kwargs.keys())}")
    
    # 打印前几个completions的内容
    # print(f"First completion: {completions if completions else 'None'}...")
    
    # 从kwargs中获取solution字段
    solutions = kwargs.get('solution', [])
    if not solutions:
        print("No solutions found, returning default rewards")
        return [0.5] * len(completions)
    
    # print(f"First solution: {solutions if solutions else 'None'}...")
    answers = solutions
    
    # 提取预测的5个字段
    pred_fields = {"description": [], "actions": [], "objects": [], "characteristics": [], "title": []}
    format_rewards=[]
    for i, completion in enumerate(completions):
        try:
            # 提取content字段
            if isinstance(completion, list) and len(completion) > 0:
                result = completion[0].get('content', '')
            elif isinstance(completion, dict):
                result = completion.get('content', '')
            else:
                result = str(completion)
            
            if result:  # 确保result不为空
                result = extract_json_from_string(result)
            else:
                result = "{}"  # 空JSON对象
            data = json.loads(result)
            pred_fields["description"].append(data.get("description", "").lower())
            pred_fields["actions"].append(str(data.get("actions", "")).lower())
            pred_fields["objects"].append(str(data.get("objects", "")).lower())
            pred_fields["characteristics"].append(str(data.get("characteristics", "")).lower())
            pred_fields["title"].append(data.get("title", "").lower())
            format_rewards.append(1.0)
        except Exception as e:
            if i == 0:  # 只打印第一个错误
                print(f"JSON parsing failed for completion {i}: {e}")
                print(f"Result type: {type(result)}, Result: {str(result)[:300]}...")
                print(f"Original completion type: {type(completion)}")
            # 解析失败时填充空字符串
            for field in pred_fields:
                pred_fields[field].append("")
            format_rewards.append(0.0)
    
    # 提取答案的5个字段
    ans_fields = {"description": [], "actions": [], "objects": [], "characteristics": [], "title": []}
    
    for result in answers:
        try:
            data = json.loads(result)
            ans_fields["description"].append(data.get("description", "").lower())
            ans_fields["actions"].append(str(data.get("actions", "")).lower())
            ans_fields["objects"].append(str(data.get("objects", "")).lower())
            ans_fields["characteristics"].append(str(data.get("characteristics", "")).lower())
            ans_fields["title"].append(data.get("title", "").lower())
        except:
            # 解析失败时填充空字符串
            for field in ans_fields:
                ans_fields[field].append("")
    
    # 计算每个字段的相似度
    field_rewards = []
    for field in ["description", "actions", "objects", "characteristics", "title"]:
        embeddings_pred = sbert_model.encode(pred_fields[field], convert_to_tensor=True)
        embeddings_ref = sbert_model.encode(ans_fields[field], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings_pred, embeddings_ref)
        field_reward = [float(cosine_scores[i][i]) for i in range(len(pred_fields[field]))]
        field_rewards.append(field_reward)
    
    # 计算5个字段的平均值作为最终reward
    rewards = []
    for i in range(len(completions)):
        avg_reward = sum(field_rewards[j][i] for j in range(5)) / 5.0
        format_reword = format_rewards[i]
        rewards.append(min(avg_reward, format_reword))
    # print(f"REWARD FUNCTION CALLED: format_rewards={format_rewards[:3]}, avg_rewards={rewards[:3]}")
    return rewards


def composite_reward(completions, field_weights=None, **kwargs):
    """
    Composite reward function combining sentence embedding similarity and BLEU score.
    
    Args:
        completions: Model completions to evaluate
        field_weights: Dict with weights for each field (description, actions, objects, characteristics, title)
                      If None, uses equal weights (0.2 each)
        **kwargs: Additional arguments
    
    Returns:
        List of reward scores for each completion
    """
    # Default equal weights if not specified
    if field_weights is None:
        field_weights = {
            "description": 0.2,
            "actions": 0.2, 
            "objects": 0.2,
            "characteristics": 0.2,
            "title": 0.2
        }

    # 从kwargs中获取solution字段
    solutions = kwargs.get('solution', [])
    if not solutions:
        print("No solutions found, returning default rewards")
        return [0.5] * len(completions)
    
    # print(f"First solution: {solutions if solutions else 'None'}...")
    answers = solutions
    
    # Extract predicted fields from completions
    pred_fields = {"description": [], "actions": [], "objects": [], "characteristics": [], "title": []}
    format_rewards = []

    # print(f"composite_reward: completions: {completions}, answers: {answers}")
    
    for i, completion in enumerate(completions):
        try:
            # Extract content field
            if isinstance(completion, list) and len(completion) > 0:
                result = completion[0].get('content', '')
            elif isinstance(completion, dict):
                result = completion.get('content', '')
            else:
                result = str(completion)
            
            if result:
                result = extract_json_from_string(result)
            else:
                result = "{}"
                
            data = json.loads(result)
            pred_fields["description"].append(data.get("description", "").lower())
            pred_fields["actions"].append(str(data.get("actions", "")).lower())
            pred_fields["objects"].append(str(data.get("objects", "")).lower())
            pred_fields["characteristics"].append(str(data.get("characteristics", "")).lower())
            pred_fields["title"].append(data.get("title", "").lower())
            format_rewards.append(1.0)
            
        except Exception as e:
            if i == 0:  # Only print first error
                print(f"JSON parsing failed for completion {i}: {e}")
            # Fill with empty strings on parse failure
            for field in pred_fields:
                pred_fields[field].append("")
            format_rewards.append(0.0)
    
    # Extract reference fields from answers
    ans_fields = {"description": [], "actions": [], "objects": [], "characteristics": [], "title": []}
    
    for result in answers:
        try:
            data = json.loads(result)
            ans_fields["description"].append(data.get("description", "").lower())
            ans_fields["actions"].append(str(data.get("actions", "")).lower())
            ans_fields["objects"].append(str(data.get("objects", "")).lower())
            ans_fields["characteristics"].append(str(data.get("characteristics", "")).lower())
            ans_fields["title"].append(data.get("title", "").lower())
        except:
            # Fill with empty strings on parse failure
            for field in ans_fields:
                ans_fields[field].append("")
    
    # Calculate composite rewards for each field
    field_rewards = []
    # bleu_weights = (0.3, 0.4, 0.3, 0.0)  # BLEU n-gram weights
    # smoother = SmoothingFunction()
    
    for field in ["description", "actions", "objects", "characteristics", "title"]:
        # 1. Sentence embedding similarity (50% weight)
        embeddings_pred = sent_embedding.encode(pred_fields[field], convert_to_tensor=True)
        embeddings_ref = sent_embedding.encode(ans_fields[field], convert_to_tensor=True)
        cosine_scores = util.cos_sim(embeddings_pred, embeddings_ref)
        embedding_rewards = [float(cosine_scores[i][i]) for i in range(len(pred_fields[field]))]
        
        # # 2. BLEU score similarity (50% weight)
        # bleu_rewards = []
        # for pred_text, ref_text in zip(pred_fields[field], ans_fields[field]):
        #     if pred_text and ref_text:  # Only calculate if both texts exist
        #         try:
        #             reference_tokens = nltk.word_tokenize(ref_text)
        #             candidate_tokens = nltk.word_tokenize(pred_text)
        #             bleu_score = sentence_bleu([reference_tokens], candidate_tokens, 
        #                                      weights=bleu_weights, 
        #                                      smoothing_function=smoother.method4)
        #             bleu_rewards.append(bleu_score)
        #         except:
        #             bleu_rewards.append(0.0)
        #     else:
        #         bleu_rewards.append(0.0)
        
        # # Combine embedding and BLEU scores (50% each)
        # composite_field_rewards = []
        # for emb_reward, bleu_reward in zip(embedding_rewards, bleu_rewards):
        #     composite_reward = 0.5 * emb_reward + 0.5 * bleu_reward
        #     composite_field_rewards.append(composite_reward)
        
        composite_field_rewards = embedding_rewards
        field_rewards.append(composite_field_rewards)
    
    # Calculate final weighted average across all fields
    final_rewards = []
    for i in range(len(completions)):
        weighted_sum = 0.0
        total_weight = 0.0
        
        for j, field in enumerate(["description", "actions", "objects", "characteristics", "title"]):
            weight = field_weights.get(field, 0.2)  # Default to 0.2 if field not in weights
            weighted_sum += field_rewards[j][i] * weight
            total_weight += weight
        
        # Normalize by total weight and apply format penalty
        if total_weight > 0:
            avg_reward = weighted_sum / total_weight
        else:
            avg_reward = 0.0
            
        format_reward = format_rewards[i]
        final_reward = min(avg_reward, format_reward)
        final_rewards.append(final_reward)
    
    return final_rewards

def format_reward_summary(completions, **kwargs):
    """Check if the Qwen model output matches a specific format."""
    # print(completions)
    # rewards = [1.0 for content in completions]
    pattern = r"{\s*\"description\":\s*\".*\"\s*,\s*\"keywords\":\s*\[.*?\]\s*,\s*\"trigger\":\s*\".*\"\s*}.*?"
    matches = [re.search(pattern, content, re.DOTALL) is not None for content in completions]
    rewards = [1.0 if match else 0.0 for match in matches]
    # print(rewards)
    return rewards

def load_multiple_datasets(data_dir="./data", max_samples_per_dataset=3000):
    """Load multiple video datasets from data directory"""
    all_data = []
    
    # Original hardcoded list (kept for reference)
    json_files = [
        "0_30_eufy_videos.json",
        "12_actionData_videos.json", 
        "kids_videos.json",
        "smarthome_videos_2.json",
        "videos_val_oops.json"
    ]

    # Alternative: single file for testing
    # json_files = [
    #     "kids_videos.json"
    # ]
    
    # # Automatically discover all JSON files in the data directory
    # import glob
    # json_pattern = os.path.join(data_dir, "*.json")
    # json_file_paths = glob.glob(json_pattern)
    
    # # Extract just the filenames from full paths
    # json_files = [os.path.basename(path) for path in json_file_paths]
    
    # # Sort for consistent order
    # json_files.sort()
    
    # print(f"Found {len(json_files)} JSON files in {os.path.abspath(data_dir)}:")
    # for json_file in json_files:
    #     print(f"  - {json_file}")
    
    for json_file in json_files:
        file_path = os.path.join(data_dir, json_file)
        print(f"file_path: {os.path.abspath(file_path)}")
        if not os.path.exists(file_path):
            print(f"Skipping {json_file}: file not found")
            continue
            
        print(f"Loading {json_file}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        dataset_data = []
        print(f"  Processing {len(annotations[:max_samples_per_dataset])} items...")
        for i, item in enumerate(annotations[:max_samples_per_dataset]):
            try:
                # Handle different data structures
                conversations = item.get('conversations', [])
                videos = item.get('videos', [])
                
                if i < 3:  # Debug first 3 items
                    print(f"    Item {i}: conversations={len(conversations)}, videos={videos}")
                
                if not conversations or not videos:
                    if i < 3:
                        print(f"    Skipping item {i}: missing conversations or videos")
                    continue
                
                # Video validation - only check existence
                valid_videos = []
                for video in videos:
                    video_path = os.path.join(data_dir, video)
                    if os.path.exists(video_path):
                        valid_videos.append(video_path)
                    else:
                        print(f"Warning: Video file not found {video_path}")
                
                if not valid_videos:
                    continue
                videos = valid_videos

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


if __name__ == "__main__":
    import argparse
    
    # Parse custom arguments first
    custom_parser = argparse.ArgumentParser()
    custom_parser.add_argument("--data_dir", type=str, default="./data", help="Directory containing training data")
    custom_parser.add_argument("--max_samples_per_dataset", type=int, default=3000, help="Maximum samples per dataset")
    # Video processing parameters
    custom_parser.add_argument("--video_fps", type=int, default=1, help="FPS for video processing")
    custom_parser.add_argument("--video_max_frames", type=int, default=16, help="Maximum frames per video")
    custom_parser.add_argument("--video_min_pixels", type=int, default=4 * 28 * 28, help="Minimum pixels for video processing")
    custom_parser.add_argument("--video_max_pixels", type=int, default=256 * 28 * 28, help="Maximum pixels for video processing")
    custom_parser.add_argument("--video_total_pixels", type=int, default=20480 * 28 * 28, help="Total pixels for video processing")
    custom_args, remaining_args = custom_parser.parse_known_args()
    
    # Parse TRL arguments
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(remaining_args)
    
    # Print all configuration parameters (only on rank 0)
    if training_args.local_rank in [-1, 0]:
        print("=" * 50)
        print("CONFIGURATION PARAMETERS")
        print("=" * 50)
        print("\n[Script Arguments]")
        for key, value in vars(script_args).items():
            print(f"  {key}: {value}")
        print("\n[Training Arguments]")
        for key, value in vars(training_args).items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
        print("\n[Model Arguments]")
        for key, value in vars(model_args).items():
            print(f"  {key}: {value}")
        print("\n[Custom Arguments (Video & Data)]")
        for key, value in vars(custom_args).items():
            print(f"  {key}: {value}")
        print("=" * 50)
    
    ################
    # Model & Processor
    ################
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation, #"eager", "flash_attention_2"
        torch_dtype=model_args.torch_dtype ,
        trust_remote_code=True,
    )

    ################
    # Dataset
    ################
    dataset = load_multiple_datasets(custom_args.data_dir, max_samples_per_dataset=custom_args.max_samples_per_dataset)
    print(f"Loaded {len(dataset)} video samples from all datasets")

    if len(dataset) == 0:
        print("ERROR: No valid samples loaded! Check annotation file and frame paths.")
        exit(1)
    
    # Print first sample for debugging
    # print(f"First sample has {len(dataset[0]['video'])} videos")
    print(f"Prompt: {dataset[0]['prompt'][0]['content'][:100]}...")

    dataset = dataset.train_test_split(test_size=500, seed=42)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    print(f"train_dataset: {len(train_dataset)}, {train_dataset[0]}")
    # print(f"eval_dataset: {len(eval_dataset)}, {eval_dataset[0]}")

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[composite_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=None,  # No PEFT for full parameter training
        # Video processing parameters
        video_fps=custom_args.video_fps,
        video_max_frames=custom_args.video_max_frames,
        video_min_pixels=custom_args.video_min_pixels,
        video_max_pixels=custom_args.video_max_pixels,
        video_total_pixels=custom_args.video_total_pixels,
    )
    
    # Freeze vision tower after trainer initialization
    if hasattr(trainer.model, 'visual'):
        for param in trainer.model.visual.parameters():
            param.requires_grad = False
        print("Frozen vision tower")
    
    # Print trainable parameters
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    if total_params > 0:
        print(f"Trainable percentage: {trainable_params/total_params*100:.2f}%")
    else:
        print("Parameters are offloaded by DeepSpeed")

    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)
