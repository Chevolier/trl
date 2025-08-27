from config import (VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS, 
                    VIDEO_TOTAL_PIXELS, 
                    IMAGE_FACTOR, FRAME_FACTOR)
from logger_setup import setup_logging
import time
from io import BytesIO
from qwen_vl_utils import smart_resize
from decord import VideoReader, cpu
import cv2
import numpy as np
from PIL import Image

logger = setup_logging()

def validate_video_file(video_bytes: bytes) -> bool:
    """
    验证视频文件是否有效
    
    Args:
        video_bytes: 视频字节数据
        
    Returns:
        bool: 是否有效
    """
    try:
        buffer = BytesIO(video_bytes)
        # 尝试创建VideoReader来验证文件
        vr = VideoReader(buffer, ctx=cpu(0))
        total_frames = len(vr)
        
        # 检查基本属性
        if total_frames <= 0:
            logger.warning("视频文件没有有效帧")
            return False
            
        # 尝试读取第一帧来验证解码能力
        try:
            first_frame = vr[0].asnumpy()
            if first_frame.size == 0:
                logger.warning("无法解码视频帧")
                return False
        except Exception as e:
            logger.warning(f"解码第一帧失败: {e}")
            return False
            
        return True
        
    except Exception as e:
        logger.warning(f"视频文件验证失败: {e}")
        return False

def read_video_decord_backend(video_bytes: bytes, nframes: int, max_retries: int = 3):
    """
    使用decord后端读取视频，带重试机制
    
    Args:
        video_bytes: 视频字节数据
        nframes: 需要提取的帧数
        max_retries: 最大重试次数
        
    Returns:
        tuple: (video, sample_fps, duration, total_frames, video_fps)
    """
    for attempt in range(max_retries):
        try:
            st = time.time()
            buffer = BytesIO(video_bytes)
            vr = VideoReader(buffer, ctx=cpu(0))

            total_frames, video_fps = len(vr), vr.get_avg_fps()
            
            # 验证基本参数
            if total_frames <= 0:
                logger.error(f"视频没有有效帧: total_frames={total_frames}")
                return None, None, None, None, None
                
            if video_fps <= 0:
                logger.warning(f"视频FPS异常: {video_fps}, 使用默认值25")
                video_fps = 25.0
            
            # 确保nframes不超过总帧数
            actual_nframes = min(nframes, total_frames)
            if actual_nframes != nframes:
                logger.warning(f"请求帧数({nframes})超过总帧数({total_frames})，调整为{actual_nframes}")
            
            # 生成帧索引，避免重复
            if total_frames == 1:
                idx = [0] * actual_nframes  # 如果只有一帧，重复使用
            else:
                idx = np.linspace(0, total_frames - 1, actual_nframes).round().astype(int).tolist()
            
            # 批量读取帧
            try:
                video = vr.get_batch(idx).asnumpy()  # (T, H, W, C) – already RGB
            except Exception as e:
                if "decode_slice_header error" in str(e) or "Frame num change" in str(e):
                    logger.warning(f"解码错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        # 尝试逐帧读取作为备选方案
                        logger.info("尝试逐帧读取...")
                        video = []
                        for frame_idx in idx:
                            try:
                                frame = vr[frame_idx].asnumpy()
                                video.append(frame)
                            except Exception as frame_e:
                                logger.warning(f"跳过损坏帧 {frame_idx}: {frame_e}")
                                # 使用前一帧或第一帧作为替代
                                if len(video) > 0:
                                    video.append(video[-1])
                                else:
                                    # 尝试读取第一帧
                                    try:
                                        first_frame = vr[0].asnumpy()
                                        video.append(first_frame)
                                    except:
                                        logger.error("无法读取任何有效帧")
                                        return None, None, None, None, None
                        
                        if len(video) > 0:
                            video = np.stack(video)
                        else:
                            continue  # 重试
                    else:
                        raise e
                else:
                    raise e
            
            sample_fps = actual_nframes / max(total_frames, 1e-6) * video_fps
            duration = total_frames / video_fps
            
            logger.debug(f"decord: total_frames={total_frames}, fps={video_fps:.2f}, "
                        f"extracted_frames={len(video)}, time={time.time() - st:.3f}s")
            
            return video, sample_fps, duration, total_frames, video_fps
            
        except Exception as e:
            logger.error(f"读取视频失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error("所有重试都失败了")
                return None, None, None, None, None
            else:
                logger.info(f"等待后重试...")
                time.sleep(0.1)  # 短暂等待后重试

    return None, None, None, None, None


def _resize_frame_cv2(frame: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """
    Resize a single frame with OpenCV (bicubic, anti‑aliasing by default).
    """
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def gen_video_qwen_vl_utils(video_bytes: bytes, nframes: int,
                            return_video_sample_fps: bool = True):

    video, sample_fps, duration, total_frames, video_fps = read_video_decord_backend(video_bytes, nframes)
    if video is None:   
        return None
    
    nframes, height, width, _ = video.shape
    min_pixels = VIDEO_MIN_PIXELS
    total_pixels = VIDEO_TOTAL_PIXELS
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05)
    )

    resized_h, resized_w = smart_resize(
        height, width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    st = time.time()
    resized = np.empty((nframes, resized_h, resized_w, 3), dtype=np.uint8)
    for i in range(nframes):
        resized[i] = _resize_frame_cv2(video[i], resized_w, resized_h)
    # logger.info(f"resize: {(height, width)} → {(resized_h, resized_w)}, "
    #             f"{nframes} frames in {time.time() - st:.3f}s")

    return {
        "video_array":   resized,            # (T, H, W, C), uint8, RGB
        "video_width":   width,
        "video_height":  height,
        "resized_width": resized_w,
        "resized_height": resized_h,
        "fps":           sample_fps,
        "duration":      duration,
        "total_frames":  total_frames,
        "video_fps":     video_fps
    }


