import json
import os
from pathlib import Path
import boto3
import time
from typing import Dict, List, Tuple

def compare_field_similarity(client, gt_value: str, gen_value: str, field_name: str) -> float:
    """使用Claude 3.7 Sonnet比较两个字段的相似度"""
    
    # 判断是否为列表字段
    is_list_field = isinstance(gt_value, list) and isinstance(gen_value, list)        
    # 根据字段类型选择评估标准
    if is_list_field:
        criteria = """
For LIST comparison:
- Score 9-10: Generated list contains all key elements from ground truth with accurate descriptions
- Score 7-8: Generated list covers most important elements but may miss some details
- Score 5-6: Generated list includes some relevant elements but misses significant content
- Score 3-4: Generated list has minimal overlap with ground truth
- Score 1-2: Generated list is mostly irrelevant or incorrect
- Score 0: No meaningful overlap or completely wrong content
"""
    else:
        criteria = """
For TEXT comparison:
- Score 9-10: Semantic equivalence with complete information coverage
- Score 7-8: High semantic similarity with minor omissions
- Score 5-6: Moderate semantic similarity with some relevant content
- Score 3-4: Low semantic similarity with limited relevant information
- Score 1-2: Minimal semantic overlap
- Score 0: No semantic similarity or completely incorrect
"""
    
    prompt = f"""
You are evaluating the quality of a multimodal model's video understanding output. Compare the similarity between the ground truth (human-annotated) and generated descriptions for the {field_name} field.

Ground Truth {field_name}:
{gt_value}

Generated {field_name}:
{gen_value}

Evaluation criteria:
{criteria}

Focus on semantic meaning and content accuracy rather than exact wording. Return only a number between 0-10.
"""
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 10,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        })
        
        response = client.invoke_model(
            modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
            body=body
        )
        
        response_body = json.loads(response['body'].read())
        score_text = response_body['content'][0]['text'].strip()
        
        try:
            score = float(score_text)
            return max(0, min(100, score))
        except ValueError:
            print(f"无法解析分数: {score_text}")
            return 0.0
            
    except Exception as e:
        print(f"调用Claude 3.5 Sonnet时出错: {e}")
        return 0.0