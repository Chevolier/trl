import json
import os
import sys
from typing import Dict, List, Any
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from translate_via_doubao import translate_json_content

def load_jsonl_file(file_path: str) -> List[Dict[str, str]]:
    """加载JSONL文件"""
    if not os.path.exists(file_path):
        return []
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"解析JSON行时出错: {e}")
                    continue
    return data

def save_jsonl_file(file_path: str, data: List[Dict[str, str]]) -> None:
    """保存JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def translate_json_value(json_value: str) -> str:
    """翻译JSON字符串中的内容"""
    if not json_value:
        return ""
    
    try:
        # 解析JSON字符串
        parsed_json = json.loads(json_value)
        
        # 翻译JSON内容
        translated_json = translate_json_content(json.dumps(parsed_json, ensure_ascii=False))
        
        return translated_json
    except Exception as e:
        print(f"翻译JSON值时出错: {e}")
        return json_value

def translate_jsonl_values(source_file: str, target_file: str):    
    # 加载源数据
    source_data = load_jsonl_file(source_file)
    if not source_data:
        print(f"错误：无法加载源文件 {source_file}")
        return
    
    print(f"源文件包含 {len(source_data)} 条记录")
    
    # 读取target_file
    if os.path.exists(target_file):
        target_data = load_jsonl_file(target_file)
        print(f"目标文件包含 {len(target_data)} 条记录")
    else:
        target_data = []
        print(f"目标文件不存在，创建空列表")
    
    # 处理数据
    process_count = 0
    
    for item in source_data:
        process_count += 1
        video_name = list(item.keys())[0]  # 获取视频文件名
        json_value = item[video_name]  # 获取JSON字符串值
        
        is_exist = False
        for target_item in target_data:
            video_name_target = list(target_item.keys())[0]
            if video_name == video_name_target:
                print(f"[{process_count}/{len(source_data)}]视频{video_name}已存在，跳过")
                is_exist = True
                break
        
        if is_exist:
            continue
        
        print(f"[{process_count}/{len(source_data)}] 翻译条目: {video_name}")
        
        # 翻译JSON值
        translated_json_value = translate_json_value(json_value)
        
        # 创建新的条目
        translated_item = {video_name: translated_json_value}
        target_data.append(translated_item)
        
        # 实时保存
        save_jsonl_file(target_file, target_data)
        
    print(f"\n翻译完成！")
    print(f"总计: {len(target_data)} 条")
    print(f"结果已保存到: {target_file}")

if __name__ == "__main__":
    source_file = "video_gpt_responses_gen.jsonl"
    target_file = source_file.replace(".jsonl", "_zh.jsonl")
    translate_jsonl_values(source_file, target_file) 