import json
import os
from pathlib import Path
import boto3
from boto3 import Session
from botocore.config import Config
from typing import Dict, List, Tuple
from compare_similarity import compare_field_similarity
from datetime import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import time
import logging
from functools import partial
import threading
from tenacity import retry, stop_after_attempt, wait_random_exponential
from tqdm import tqdm

# AWS Bedrock 配置
config = Config(
    region_name='us-east-1',
    signature_version='v4',
    retries={'max_attempts': 10, 'mode': 'standard'}
)

# 带重试机制的字段相似度比较函数
@retry(stop=stop_after_attempt(5), wait=wait_random_exponential(multiplier=1, max=3))
def compare_field_similarity_with_retry(gt_value: str, gen_value: str, field_name: str) -> float:
    """使用Claude 3.7 Sonnet比较两个字段的相似度，带重试机制"""
    
    # 创建新的客户端实例（避免多进程共享问题）
    client = boto3.client('bedrock-runtime', config=config)
    
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

def load_jsonl_file(file_path: str) -> Dict[str, str]:
    """
    读取JSONL文件并解析成字典
    
    Args:
        file_path (str): JSONL文件路径
    
    Returns:
        Dict[str, str]: 包含video_name和JSON字符串的字典
    """
    result_dict = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                # 获取第一个（也是唯一的）键值对
                video_name = list(data.keys())[0]
                json_str = data[video_name]
                result_dict[video_name] = json_str
                
            except json.JSONDecodeError as e:
                print(f"解析JSON行时出错: {e}")
                continue
            except Exception as e:
                print(f"处理行时出错: {e}")
                continue
    
    return result_dict

def parse_json_string(json_str: str) -> Dict[str, Any]:
    """
    解析JSON字符串，提取各个字段
    
    Args:
        json_str (str): JSON字符串或字典，可能包含代码块标记
    
    Returns:
        Dict[str, Any]: 解析后的字段字典
    """
    try:
        # 如果已经是字典类型，直接使用
        if isinstance(json_str, dict):
            data = json_str
        else:
            # 清理字符串：移除代码块标记和多余的空白字符
            cleaned_str = clean_json_string(json_str)
            # 尝试解析JSON字符串
            data = json.loads(cleaned_str)
        
        return {
            'description': data.get('description', ''),
            'actions': data.get('actions', []),
            'objects': data.get('objects', []),
            'characteristics': data.get('characteristics', []),
            'title': data.get('title', '')
        }
    except json.JSONDecodeError as e:
        print(f"解析JSON字符串时出错: {e}")
        print(f"原始字符串: {json_str[:200]}...")
        return {
            'description': '',
            'actions': [],
            'objects': [],
            'characteristics': [],
            'title': ''
        }
    except Exception as e:
        print(f"解析数据时出错: {e}")
        return {
            'description': '',
            'actions': [],
            'objects': [],
            'characteristics': [],
            'title': ''
        }

def clean_json_string(json_str: str) -> str:
    """
    清理JSON字符串，移除代码块标记和多余的格式
    
    Args:
        json_str (str): 原始JSON字符串
    
    Returns:
        str: 清理后的JSON字符串
    """
    # 移除代码块标记 ```json 和 ```
    cleaned = re.sub(r'```json\s*', '', json_str)
    cleaned = re.sub(r'```\s*\$', '', cleaned)
    
    # 移除字符串开头和结尾的空白字符
    cleaned = cleaned.strip()
    
    # 如果字符串以 "{ 开始，移除开头的引号
    if cleaned.startswith('"') and cleaned.endswith('"'):
        cleaned = cleaned[1:-1]
        # 处理转义的引号
        cleaned = cleaned.replace('\\"', '"')
    
    return cleaned

# 单个视频处理函数（用于多进程）
def process_single_video(video_data):
    """
    处理单个视频的评估任务
    
    Args:
        video_data: 包含视频信息的元组 (video_name, gt_data, gen_data, gt_data_zh, gen_data_zh)
    
    Returns:
        Dict: 评估结果记录
    """
    video_name, gt_data, gen_data, gt_data_zh, gen_data_zh = video_data
    
    try:
        scores = {}
        
        # 比较各个字段
        fields = ['description', 'actions', 'objects', 'characteristics', 'title']
        
        for field in fields:
            gt_value = str(gt_data.get(field, ''))
            gen_value = str(gen_data.get(field, ''))
            scores[field] = compare_field_similarity_with_retry(gt_value, gen_value, field)
        
        # 构建结果记录
        score_record = {
            "video_name": video_name,
            "avg_score": sum(scores.values()) / len(scores),
            "scores": scores,
            "gen": json.dumps(gen_data, ensure_ascii=False),
            "gt": json.dumps(gt_data, ensure_ascii=False),
            "gen_zh": json.dumps(gen_data_zh, ensure_ascii=False),
            "gt_zh": json.dumps(gt_data_zh, ensure_ascii=False)
        }
        
        return score_record
        
    except Exception as e:
        print(f"[Error] 处理视频 {video_name} 时出错: {e}")
        return None

def main(output_dir = "result"):
    # 设置文件路径
    current_dir = Path(__file__).parent
    gt_file = "./video_gpt_responses_gt.jsonl"
    gen_file = "/home/ec2-user/lumi_project/predict_results/video_gpt_responses_gen_32b.jsonl"
    gt_file_zh = gt_file.replace(".jsonl", "_zh.jsonl")
    gen_file_zh = gen_file.replace(".jsonl", "_zh.jsonl")
    print(f'gt_file is {gt_file}, gt_file_zh is {gt_file_zh}')
    print(f'gen_file is {gen_file}, gen_file_zh is {gen_file_zh}')
    
    # 检查文件是否存在
    if not os.path.exists(gt_file):
        print(f"错误: GT文件不存在: {gt_file}")
        return
    
    if not os.path.exists(gen_file):
        print(f"错误: GEN文件不存在: {gen_file}")
        return
    
    if not os.path.exists(gt_file_zh):
        print(f"错误: GT中文文件不存在: {gt_file_zh}")
        return
    
    if not os.path.exists(gen_file_zh):
        print(f"错误: GEN中文文件不存在，跳过: {gen_file_zh}")
        gen_file_zh = gen_file
            
    # 初始化AWS Bedrock客户端
    client = boto3.client(
        'bedrock-runtime',
        region_name='us-east-1'
    )
    
    # 读取GT和GEN数据
    print("正在读取GT数据...")
    gt_dict = load_jsonl_file(gt_file)
    print(f"GT数据读取完成，共 {len(gt_dict)} 个视频")
    gt_dict_zh = load_jsonl_file(gt_file_zh)
    print(f"GT中文数据读取完成，共 {len(gt_dict_zh)} 个视频")
    
    print("正在读取GEN数据...")
    gen_dict = load_jsonl_file(gen_file)
    print(f"GEN数据读取完成，共 {len(gen_dict)} 个视频")
    gen_dict_zh = load_jsonl_file(gen_file_zh)
    print(f"GEN中文数据读取完成，共 {len(gen_dict_zh)} 个视频")
    
    # 找到共同的视频
    common_videos = set(gt_dict.keys()) & set(gen_dict.keys())
    print(f"找到 {len(common_videos)} 个共同视频")
    
    if len(common_videos) == 0:
        print("错误: 没有找到共同的视频")
        return
    
    # 准备多进程处理的数据
    video_data_list = []
    for i, video_name in enumerate(common_videos):
        if i < 500:  # 限制处理数量
            continue
            
        # 解析GT和GEN数据
        gt_data = parse_json_string(gt_dict[video_name])
        gen_data = parse_json_string(gen_dict[video_name])
        gt_data_zh = parse_json_string(gt_dict_zh[video_name])
        gen_data_zh = parse_json_string(gen_dict_zh[video_name])
        
        video_data_list.append((video_name, gt_data, gen_data, gt_data_zh, gen_data_zh))
    
    print(f"准备使用多进程处理 {len(video_data_list)} 个视频...")
    
    # 使用多进程处理
    score_records = []
    max_workers = 4  # 可以根据需要调整并发数
    
    # 写入锁（避免线程写入冲突）
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_video, video_data) 
                  for video_data in video_data_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            result = future.result()
            if result is not None:
                with lock:
                    score_records.append(result)
                    print(f"完成处理视频: {result['video_name']}, 平均分数: {result['avg_score']:.2f}")
                    print(result)
    
    print(f"多进程处理完成，成功处理 {len(score_records)} 个视频")
        
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cur_time = datetime.now().strftime('%Y%m%d%H%M')
    gen_file_name = Path(gen_file).stem
    markdown_file = output_dir / f"eval_result_{gen_file_name}_{cur_time}_32B_check.md"
    
    # 直接生成Markdown报告
    generate_evaluation_report(score_records, markdown_file)
    print(f"Markdown报告已保存到: {markdown_file}")

def highlight_content(content: str):
    highlight_list = ["description", "actions", "objects", "characteristics", "title"]
    for hl in highlight_list:
        content = content.replace(f"{hl}", f"**{hl}**")
    return content

def generate_evaluation_report(score_records, output_file):
    """
    直接生成评估报告Markdown文件
    """
    # 计算统计信息
    total_videos = len(score_records)
    
    # 计算各字段的平均分数
    field_scores = {
        'description': [],
        'actions': [],
        'objects': [],
        'characteristics': [],
        'title': []
    }
    
    for record in score_records:
        scores = record.get('scores', {})
        for field in field_scores.keys():
            if field in scores:
                field_scores[field].append(scores[field])
    
    # 计算平均分数
    avg_scores = {}
    for field, scores in field_scores.items():
        if scores:
            avg_scores[field] = sum(scores) / len(scores)
        else:
            avg_scores[field] = 0.0
    
    # 计算总体平均分数
    all_scores = []
    for scores in field_scores.values():
        all_scores.extend(scores)
    
    overall_avg = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    # 生成Markdown内容
    markdown_content = []
    
    # 添加标题
    cur_time = datetime.now().strftime('%Y%m%d-%H%M')
    markdown_content.append(f'## 版本时间：{cur_time}')
    markdown_content.append('')
    markdown_content.append('## 评估结果概览')
    markdown_content.append('')
    
    # 添加总体统计
    markdown_content.append('### 总体统计')
    markdown_content.append(f'- **评估视频总数：** `{total_videos}` 个')
    markdown_content.append(f'- **总体平均相似度：** `{overall_avg:.2f}`')
    markdown_content.append('')
    
    # 添加各字段平均相似度表格
    markdown_content.append('### 各字段平均相似度')
    markdown_content.append('')
    markdown_content.append('| 字段 | 平均相似度 |')
    markdown_content.append('|------|------------|')
    markdown_content.append(f'| **Description** | `{avg_scores["description"]:.3f}` |')
    markdown_content.append(f'| **Objects** | `{avg_scores["objects"]:.2f}` |')
    markdown_content.append(f'| **Actions** | `{avg_scores["actions"]:.2f}` |')
    markdown_content.append(f'| **Characteristics** | `{avg_scores["characteristics"]:.2f}` |')
    markdown_content.append(f'| **Title** | `{avg_scores["title"]:.2f}` |')
    markdown_content.append('')
    
    # 添加详细评分表格
    markdown_content.append('## 详细评分结果')
    markdown_content.append('')
    markdown_content.append(f'| 视频名称 | 平均分数 | Description | Actions | Objects | Characteristics | Title | Human Annotation  | {cur_time} |')
    markdown_content.append('|:--------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:---------------:|:---------------:|')
    
    # 按平均分数排序
    score_records.sort(key=lambda x: x.get('avg_score', 0), reverse=True)
    
    markdown_content_zh = markdown_content.copy()
    
    for record in score_records:
        video_name = record.get('video_name', '').replace("_", "/")
        video_url = f"[{video_name}](https://alab-cos-1300889962.cos.ap-beijing.myqcloud.com/alab_openai_prod/video/{video_name})"
        avg_score = record.get('avg_score', 0)
        scores = record.get('scores', {})
        gt = record.get('gt', "")
        gen = record.get('gen', "")
        gt_zh = record.get('gt_zh', "")
        gen_zh = record.get('gen_zh', "")
        gt = highlight_content(gt)
        gen = highlight_content(gen)
        gt_zh = highlight_content(gt_zh)
        gen_zh = highlight_content(gen_zh)
        
        desc_score = scores.get('description', 0)
        actions_score = scores.get('actions', 0)
        objects_score = scores.get('objects', 0)
        characteristics_score = scores.get('characteristics', 0)
        title_score = scores.get('title', 0)
        
        markdown_content.append(f'| {video_url} | **{avg_score:.1f}** | {desc_score} | {actions_score} | {objects_score} | {characteristics_score} | {title_score} |{gt} | {gen} |')
        markdown_content_zh.append(f'| {video_url} | **{avg_score:.1f}** | {desc_score} | {actions_score} | {objects_score} | {characteristics_score} | {title_score} |{gt_zh} | {gen_zh} |')
    
    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_content))
    output_file_zh = str(output_file).replace(".md", "_zh.md")
    with open(output_file_zh, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_content_zh))
    
    print(f'评估报告已生成: {output_file}, 中文报告已生成: {output_file_zh}')

if __name__ == "__main__":
    main()
