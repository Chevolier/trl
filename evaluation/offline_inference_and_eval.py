from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from config import USER_PROMPT_TEMPLATE, NFRAMES, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS
from video_util import gen_video_qwen_vl_utils
from logger_setup import setup_logging
import os
import json
import time
import argparse
from typing import List, Tuple, Dict
from pathlib import Path
from datetime import datetime

# Import evaluation functions
from run_eval import (
    load_jsonl_file, 
    parse_json_string, 
    process_single_video, 
    generate_evaluation_report
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

logger = setup_logging()

def prepare_video_batch(video_files: List[Tuple[str, str]], processor) -> Tuple[List[Dict], List[str]]:
    """批量准备视频数据"""
    batch_inputs = []
    batch_keys = []
    
    for video_path, key in video_files:
        try:
            # 读取视频文件
            with open(video_path, "rb") as f:
                video_bytes = f.read()
            
            # 处理视频数据
            video_inputs = gen_video_qwen_vl_utils(
                video_bytes, 
                nframes=NFRAMES, 
                return_video_sample_fps=False
            )
            
            if video_inputs is None:
                logger.error(f"Failed to process video: {video_path}")
                continue
            
            # 构建消息
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": USER_PROMPT_TEMPLATE},
                    {
                        "type": "video",
                        "video": video_inputs["video_array"],
                        "max_pixels": VIDEO_MAX_PIXELS,
                        "min_pixels": VIDEO_MIN_PIXELS
                    }
                ],
            }]
            
            # 使用processor生成prompt
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            # 准备多模态数据
            mm_data = {}
            if video_inputs is not None and "video_array" in video_inputs:
                mm_data["video"] = video_inputs["video_array"]
            
            # 准备输入
            llm_input = {
                "prompt": prompt,
                "multi_modal_data": mm_data,
            }
            
            batch_inputs.append(llm_input)
            batch_keys.append(key)
            
        except Exception as e:
            logger.error(f"Error preparing video {video_path}: {e}")
            continue
    
    return batch_inputs, batch_keys

def generate_batch_via_vllm_offline(llm, processor, video_files: List[Tuple[str, str]], batch_size: int = 4):
    """使用离线VLLM批量生成视频分析结果"""
    all_results = {}
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.01,
        max_tokens=512
    )
    
    # 计算总批次数
    total_batches = (len(video_files) + batch_size - 1) // batch_size
    
    # 使用tqdm显示进度条
    with tqdm(total=len(video_files), desc="推理进度", unit="videos") as pbar:
        # 分批处理
        for i in range(0, len(video_files), batch_size):
            batch_files = video_files[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            try:
                # 准备批量数据
                batch_inputs, batch_keys = prepare_video_batch(batch_files, processor)
                
                if not batch_inputs:
                    logger.warning(f"批次 {batch_num}/{total_batches} 无有效输入")
                    pbar.update(len(batch_files))
                    continue
                
                # 批量推理
                outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
                
                # 处理结果
                for j, output in enumerate(outputs):
                    if j < len(batch_keys) and output.outputs:
                        key = batch_keys[j]
                        result = output.outputs[0].text
                        all_results[key] = result
                    else:
                        logger.error(f"批次 {batch_num} 第 {j} 项无输出")
                
                # 更新进度条
                pbar.update(len(batch_inputs))
                        
            except Exception as e:
                logger.error(f"批次 {batch_num}/{total_batches} 处理错误: {e}")
                pbar.update(len(batch_files))
                continue
    
    return all_results

def run_evaluation(gen_file: str, gt_file: str = "./video_gpt_responses_gt.jsonl", 
                  output_dir: str = "result", max_videos: int = 500):
    """
    运行评估并生成markdown报告
    
    Args:
        gen_file: 生成的结果文件路径
        gt_file: 真实标签文件路径
        output_dir: 输出目录
        max_videos: 最大评估视频数量
    
    Returns:
        tuple: (markdown文件路径, 评估总时间, 平均单条评估时间)
    """
    logger.info("开始运行评估...")
    eval_start_time = time.time()
    
    # 设置文件路径
    gt_file_zh = gt_file.replace(".jsonl", "_zh.jsonl")
    gen_file_zh = gen_file.replace(".jsonl", "_zh.jsonl")
    
    logger.info(f'GT文件: {gt_file}, GT中文文件: {gt_file_zh}')
    logger.info(f'生成文件: {gen_file}, 生成中文文件: {gen_file_zh}')
    
    # 检查文件是否存在
    if not os.path.exists(gt_file):
        logger.error(f"GT文件不存在: {gt_file}")
        return None, 0, 0
    
    if not os.path.exists(gen_file):
        logger.error(f"生成文件不存在: {gen_file}")
        return None, 0, 0
    
    if not os.path.exists(gt_file_zh):
        logger.warning(f"GT中文文件不存在: {gt_file_zh}")
        gt_file_zh = gt_file
    
    if not os.path.exists(gen_file_zh):
        logger.warning(f"生成中文文件不存在，使用原文件: {gen_file_zh}")
        gen_file_zh = gen_file
    
    # 读取数据
    logger.info("正在读取GT数据...")
    gt_dict = load_jsonl_file(gt_file)
    logger.info(f"GT数据读取完成，共 {len(gt_dict)} 个视频")
    
    gt_dict_zh = load_jsonl_file(gt_file_zh)
    logger.info(f"GT中文数据读取完成，共 {len(gt_dict_zh)} 个视频")
    
    logger.info("正在读取生成数据...")
    gen_dict = load_jsonl_file(gen_file)
    logger.info(f"生成数据读取完成，共 {len(gen_dict)} 个视频")
    
    gen_dict_zh = load_jsonl_file(gen_file_zh)
    logger.info(f"生成中文数据读取完成，共 {len(gen_dict_zh)} 个视频")
    
    # 找到共同的视频
    common_videos = set(gt_dict.keys()) & set(gen_dict.keys())
    logger.info(f"找到 {len(common_videos)} 个共同视频")
    
    if len(common_videos) == 0:
        logger.error("没有找到共同的视频")
        return None, 0, 0
    
    # 准备多进程处理的数据
    video_data_list = []
    for i, video_name in enumerate(common_videos):
        if i >= max_videos:  # 限制处理数量
            break
            
        # 解析GT和GEN数据
        gt_data = parse_json_string(gt_dict[video_name])
        gen_data = parse_json_string(gen_dict[video_name])
        gt_data_zh = parse_json_string(gt_dict_zh.get(video_name, gt_dict[video_name]))
        gen_data_zh = parse_json_string(gen_dict_zh.get(video_name, gen_dict[video_name]))
        
        video_data_list.append((video_name, gt_data, gen_data, gt_data_zh, gen_data_zh))
    
    logger.info(f"准备使用多进程处理 {len(video_data_list)} 个视频...")
    
    # 使用多进程处理
    score_records = []
    max_workers = 4  # 可以根据需要调整并发数
    
    # 写入锁（避免线程写入冲突）
    lock = threading.Lock()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_video, video_data) 
                  for video_data in video_data_list]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="评估进度"):
            result = future.result()
            if result is not None:
                with lock:
                    score_records.append(result)
    
    eval_end_time = time.time()
    eval_total_time = eval_end_time - eval_start_time
    
    # 评估统计信息
    if len(score_records) > 0:
        avg_eval_time = eval_total_time / len(score_records)
        logger.info("=" * 50)
        logger.info("评估完成统计:")
        logger.info(f"- 评估总耗时: {eval_total_time:.2f}秒")
        logger.info(f"- 评估视频数量: {len(score_records)}")
        logger.info(f"- 平均单条评估时间: {avg_eval_time:.3f}秒")
        logger.info("=" * 50)
    else:
        logger.error("评估失败，没有成功处理的视频")
        return None, eval_total_time, 0
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cur_time = datetime.now().strftime('%Y%m%d%H%M')
    gen_file_name = Path(gen_file).stem
    markdown_file = output_dir / f"eval_result_{gen_file_name}_{cur_time}.md"
    
    # 生成Markdown报告
    generate_evaluation_report(score_records, markdown_file)
    logger.info(f"Markdown报告已保存到: {markdown_file}")
    
    avg_eval_time = eval_total_time / len(score_records) if len(score_records) > 0 else 0
    return str(markdown_file), eval_total_time, avg_eval_time

def run_inference(model_path: str, analyse_dir: str, predict_file: str, batch_size: int = 4):
    """
    执行推理流程
    
    Args:
        model_path: 模型路径
        analyse_dir: 视频分析目录
        predict_file: 预测结果文件路径
        batch_size: 批处理大小
    
    Returns:
        tuple: (推理结果数量, 推理总时间, 平均单条推理时间)
    """
    logger.info("=" * 60)
    logger.info("开始推理阶段")
    logger.info("=" * 60)
    
    # 删除已存在的输出文件
    if os.path.exists(predict_file):
        os.remove(predict_file)
        logger.info(f"删除已存在的预测文件: {predict_file}")
    
    # 初始化VLLM模型
    logger.info(f"Loading VLLM model from {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        # tensor_parallel_size=1,  # 取消注释来指定使用的GPU数量
        # gpu_ids=[0],  # 取消注释来指定使用特定的GPU ID
    )
    
    # 初始化processor
    processor = AutoProcessor.from_pretrained(model_path)
    logger.info("VLLM model loaded successfully")
    
    # 收集所有视频文件
    video_files = []
    for group_id in os.listdir(analyse_dir):
        group_path = os.path.join(analyse_dir, group_id)
        if not os.path.isdir(group_path):
            continue
            
        for video_name in os.listdir(group_path):
            video_path = os.path.join(group_path, video_name)
            key = f'{group_id}_{video_name}'
            video_files.append((video_path, key))

    # video_files=video_files[0:10]
    
    logger.info(f"Found {len(video_files)} videos to process")
    
    # 批量处理所有视频
    logger.info("开始推理...")
    inference_start_time = time.time()
    all_results = generate_batch_via_vllm_offline(llm, processor, video_files, batch_size)
    inference_end_time = time.time()
    inference_total_time = inference_end_time - inference_start_time
    
    # 推理统计信息
    if len(all_results) > 0:
        avg_inference_time = inference_total_time / len(all_results)
        logger.info("=" * 50)
        logger.info("推理完成统计:")
        logger.info(f"- 推理总耗时: {inference_total_time:.2f}秒")
        logger.info(f"- 推理视频数量: {len(all_results)}")
        logger.info(f"- 平均单条推理时间: {avg_inference_time:.3f}秒")
        logger.info(f"- 推理吞吐量: {len(all_results)/inference_total_time:.2f} videos/s")
        logger.info("=" * 50)
        
        # 保存所有结果
        with open(predict_file, 'w', encoding='utf-8') as f:
            for key, result in all_results.items():
                json_line = json.dumps({key: result}, ensure_ascii=False) + '\n'
                f.write(json_line)
        
        logger.info(f"推理结果已保存到: {predict_file}")
        return len(all_results), inference_total_time, avg_inference_time
    else:
        logger.error("推理失败，没有成功处理的视频")
        return 0, inference_total_time, 0



def main(model_path: str, analyse_dir: str, gt_file: str, batch_size: int = 4, 
         do_inference: bool = True, do_evaluation: bool = True, predict_file: str = None, predict_dir: str = "../predict_results"):
    """
    主函数：执行推理和评估流程
    
    Args:
        model_path: 模型路径
        analyse_dir: 视频分析目录
        gt_file: 真实标签文件路径
        batch_size: 批处理大小
        do_inference: 是否执行推理
        do_evaluation: 是否执行评估
        predict_file: 预测文件路径（可选）
    """
    # 从模型路径提取模型名称
    model_name = Path(model_path).name if model_path else "unknown_model"
    if predict_file is None:
        # 确保predict_dir目录存在
        predict_dir_path = Path(predict_dir)
        predict_dir_path.mkdir(parents=True, exist_ok=True)
        predict_file = str(predict_dir_path / f"{model_name}_predict.jsonl")
    
    logger.info("=" * 60)
    logger.info("任务配置:")
    logger.info(f"- 执行推理: {do_inference}")
    logger.info(f"- 执行评估: {do_evaluation}")
    if do_inference:
        logger.info(f"- 模型路径: {model_path}")
        logger.info(f"- 视频目录: {analyse_dir}")
        logger.info(f"- 批处理大小: {batch_size}")
    if do_evaluation:
        logger.info(f"- GT文件: {gt_file}")
    logger.info(f"- 预测文件: {predict_file}")
    logger.info("=" * 60)
    
    # 初始化统计变量
    inference_count = 0
    inference_total_time = 0
    avg_inference_time = 0
    eval_total_time = 0
    avg_eval_time = 0
    markdown_file = None
    
    # 执行推理
    if do_inference:
        # 验证推理参数
        if not os.path.exists(model_path):
            logger.error(f"模型路径不存在: {model_path}")
            return
        
        if not os.path.exists(analyse_dir):
            logger.error(f"视频目录不存在: {analyse_dir}")
            return
        
        inference_count, inference_total_time, avg_inference_time = run_inference(
            model_path, analyse_dir, predict_file, batch_size
        )
        
        if inference_count == 0:
            logger.error("推理失败")
            if do_evaluation:
                logger.error("推理失败，跳过评估步骤")
                return
    
    # 执行评估
    if do_evaluation:
        # 验证评估参数
        if not os.path.exists(predict_file):
            logger.error(f"预测文件不存在: {predict_file}")
            return
        
        if not os.path.exists(gt_file):
            logger.error(f"GT文件不存在: {gt_file}")
            return
        
        eval_output_dir = "eval_results"
        max_eval_videos = 1000
        
        try:
            markdown_file, eval_total_time, avg_eval_time = run_evaluation(
                predict_file, gt_file, eval_output_dir, max_eval_videos
            )
            
            if markdown_file:
                logger.info(f"评估完成！Markdown报告已保存到: {markdown_file}")
            else:
                logger.error("评估失败，请检查GT文件和推理结果文件")
                
        except Exception as e:
            logger.error(f"运行评估时出错: {e}")
            logger.exception("评估错误详情:")
            return
    
    # 最终统计信息
    logger.info("=" * 60)
    logger.info("任务完成总结:")
    logger.info(f"- 模型名称: {model_name}")
    
    if do_inference:
        logger.info(f"- 推理处理视频数量: {inference_count}")
        logger.info(f"- 推理总耗时: {inference_total_time:.2f}秒")
        logger.info(f"- 平均单条推理时间: {avg_inference_time:.3f}秒")
        if inference_total_time > 0:
            logger.info(f"- 推理吞吐量: {inference_count/inference_total_time:.2f} videos/s")
        logger.info(f"- 推理结果文件: {predict_file}")
    
    if do_evaluation:
        logger.info(f"- 评估总耗时: {eval_total_time:.2f}秒")
        logger.info(f"- 平均单条评估时间: {avg_eval_time:.3f}秒")
        if markdown_file:
            logger.info(f"- 评估报告文件: {markdown_file}")
    
    if do_inference and do_evaluation:
        total_time = inference_total_time + eval_total_time
        logger.info(f"- 总耗时: {total_time:.2f}秒")
    
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线视频推理和评估脚本")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/home/ec2-user/lumi_project2/LLaMA-Factory/model_qwen_7b_512_pixels_64_smarthome_2_step/checkpoint-20",
        help="模型路径"
    )
    parser.add_argument(
        "--analyse_dir", 
        type=str, 
        default="/home/ec2-user/lumi_project2/LLaMA-Factory/data/group_stand_revised",
        help="视频分析目录"
    )
    parser.add_argument(
        "--gt_file", 
        type=str, 
        default="./video_gpt_responses_gt.jsonl",
        help="真实标签文件路径 (默认: ./video_gpt_responses_gt.jsonl)"
    )
    parser.add_argument(
        "--predict_file", 
        type=str, 
        default=None,
        help="预测结果文件路径 (可选，默认根据模型名称生成)"
    )
    parser.add_argument(
        "--predict_dir", 
        type=str, 
        default="../predict_results",
        help="预测结果文件目录 (默认: ../predict_results)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="批处理大小 (默认: 16)"
    )
    parser.add_argument(
        "--skip_inference", 
        action="store_true", 
        default=False,
        help="跳过推理步骤 (默认: False)"
    )
    parser.add_argument(
        "--skip_evaluation", 
        action="store_true", 
        default=False,
        help="跳过评估步骤 (默认: False)"
    )
    
    args = parser.parse_args()
    
    # 处理推理和评估标志
    do_inference = not args.skip_inference
    do_evaluation = not args.skip_evaluation
    
    # 验证参数
    if do_inference:
        if not os.path.exists(args.model_path):
            logger.error(f"模型路径不存在: {args.model_path}")
            exit(1)
        
        if not os.path.exists(args.analyse_dir):
            logger.error(f"视频目录不存在: {args.analyse_dir}")
            exit(1)
    
    if do_evaluation and not os.path.exists(args.gt_file):
        logger.error(f"GT文件不存在: {args.gt_file}")
        exit(1)
    
    if not do_inference and not do_evaluation:
        logger.error("必须至少选择执行推理或评估中的一项")
        exit(1)
    
    # 运行主函数
    main(
        model_path=args.model_path, 
        analyse_dir=args.analyse_dir, 
        gt_file=args.gt_file,
        batch_size=args.batch_size, 
        do_inference=do_inference, 
        do_evaluation=do_evaluation,
        predict_file=args.predict_file,
        predict_dir=args.predict_dir
    )

