from logger_setup import setup_logging
import time
from io import BytesIO
from qwen_vl_utils import smart_resize
from decord import VideoReader, cpu
import cv2
import numpy as np
from PIL import Image
from config import VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS, NFRAMES, IMAGE_FACTOR, FRAME_FACTOR

logger = setup_logging()

# 从config.py导入的参数
TRAINING_VIDEO_MAX_PIXELS = VIDEO_MAX_PIXELS
TRAINING_VIDEO_MIN_PIXELS = VIDEO_MIN_PIXELS
TRAINING_VIDEO_MAXLEN = NFRAMES
TRAINING_VIDEO_FPS = 2.0  # 训练时的video_fps

def validate_video_file(video_bytes: bytes) -> bool:
    """验证视频文件是否有效"""
    try:
        buffer = BytesIO(video_bytes)
        vr = VideoReader(buffer, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames <= 0:
            logger.warning("视频文件没有有效帧")
            return False
            
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

def read_video_training_aligned(video_bytes: bytes, nframes: int = TRAINING_VIDEO_MAXLEN, max_retries: int = 3):
    """
    按照训练时的逻辑读取视频
    
    Args:
        video_bytes: 视频字节数据
        nframes: 需要提取的帧数，默认使用训练时的64帧
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
            
            if total_frames <= 0:
                logger.error(f"视频没有有效帧: total_frames={total_frames}")
                return None, None, None, None, None
                
            if video_fps <= 0:
                logger.warning(f"视频FPS异常: {video_fps}, 使用默认值25")
                video_fps = 25.0
            
            # 按照训练时的逻辑进行帧采样
            # 训练时使用的是均匀采样策略
            actual_nframes = min(nframes, total_frames)
            if actual_nframes != nframes:
                logger.debug(f"请求帧数({nframes})超过总帧数({total_frames})，调整为{actual_nframes}")
            
            # 均匀采样帧索引 - 与训练时保持一致
            if total_frames == 1:
                idx = [0] * actual_nframes
            else:
                idx = np.linspace(0, total_frames - 1, actual_nframes).round().astype(int).tolist()
            
            # 批量读取帧
            try:
                video = vr.get_batch(idx).asnumpy()  # (T, H, W, C) RGB格式
                
                # Qwen2VL特殊处理：确保帧数为偶数
                if len(video) % 2 != 0:
                    video = np.concatenate([video, video[-1:]], axis=0)
                    actual_nframes = len(video)
            except Exception as e:
                if "decode_slice_header error" in str(e) or "Frame num change" in str(e):
                    logger.warning(f"解码错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                    if attempt < max_retries - 1:
                        # 逐帧读取作为备选方案
                        video = []
                        for frame_idx in idx:
                            try:
                                frame = vr[frame_idx].asnumpy()
                                video.append(frame)
                            except Exception as frame_e:
                                logger.warning(f"跳过损坏帧 {frame_idx}: {frame_e}")
                                if len(video) > 0:
                                    video.append(video[-1])  # 使用前一帧
                                else:
                                    try:
                                        first_frame = vr[0].asnumpy()
                                        video.append(first_frame)
                                    except:
                                        logger.error("无法读取任何有效帧")
                                        return None, None, None, None, None
                        
                        if len(video) > 0:
                            video = np.stack(video)
                        else:
                            continue
                    else:
                        raise e
                else:
                    raise e
            
            # 计算采样FPS - 与训练时保持一致
            sample_fps = actual_nframes / max(total_frames, 1e-6) * video_fps
            duration = total_frames / video_fps
            
            logger.debug(f"视频读取: total_frames={total_frames}, fps={video_fps:.2f}, "
                        f"extracted_frames={len(video)}, time={time.time() - st:.3f}s")
            
            return video, sample_fps, duration, total_frames, video_fps
            
        except Exception as e:
            logger.error(f"读取视频失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error("所有重试都失败了")
                return None, None, None, None, None
            else:
                time.sleep(0.1)

    return None, None, None, None, None

def _preprocess_frame_qwen2vl(frame: np.ndarray) -> np.ndarray:
    """Qwen2VL特殊的帧预处理逻辑，与训练时保持一致"""
    height, width = frame.shape[:2]
    
    # 确保最小尺寸至少28像素
    if min(width, height) < 28:
        width, height = max(width, 28), max(height, 28)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # 防止极端宽高比 (>200)
    if width / height > 200:
        width, height = int(height * 180), height
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    
    if height / width > 200:
        width, height = width, int(width * 180)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
    
    return frame

def _resize_frame_cv2(frame: np.ndarray, new_w: int, new_h: int) -> np.ndarray:
    """使用OpenCV调整帧大小"""
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

def gen_video_qwen_vl_utils_training_aligned(video_bytes: bytes, 
                                           nframes: int = TRAINING_VIDEO_MAXLEN,
                                           return_video_sample_fps: bool = True):
    """
    按照训练时的逻辑处理视频
    
    Args:
        video_bytes: 视频字节数据
        nframes: 帧数，默认使用训练时的64帧
        return_video_sample_fps: 是否返回采样FPS
        
    Returns:
        dict: 包含处理后视频数据的字典
    """
    
    video, sample_fps, duration, total_frames, video_fps = read_video_training_aligned(video_bytes, nframes)
    if video is None:   
        return None
    
    nframes, height, width, _ = video.shape
    
    # 使用评估时的像素限制参数 - 与eval/config.py保持一致
    min_pixels = TRAINING_VIDEO_MIN_PIXELS        # 与eval/config.py的VIDEO_MIN_PIXELS一致
    max_pixels = TRAINING_VIDEO_MAX_PIXELS        # 与eval/config.py的VIDEO_MAX_PIXELS一致
    
    # 按照训练时的逻辑计算resize参数
    resized_h, resized_w = smart_resize(
        height, width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    
    # 调整所有帧的大小 - 先进行Qwen2VL预处理，再resize
    st = time.time()
    resized = np.empty((nframes, resized_h, resized_w, 3), dtype=np.uint8)
    for i in range(nframes):
        # 先进行Qwen2VL特殊预处理
        preprocessed_frame = _preprocess_frame_qwen2vl(video[i])
        # 再进行最终resize
        resized[i] = _resize_frame_cv2(preprocessed_frame, resized_w, resized_h)
    
    logger.debug(f"resize: {(height, width)} → {(resized_h, resized_w)}, "
                f"{nframes} frames in {time.time() - st:.3f}s")

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

# 为了兼容性，提供原函数名的别名
def gen_video_qwen_vl_utils(video_bytes: bytes, nframes: int = TRAINING_VIDEO_MAXLEN,
                           return_video_sample_fps: bool = True):
    """兼容性函数，调用训练对齐的版本"""
    return gen_video_qwen_vl_utils_training_aligned(video_bytes, nframes, return_video_sample_fps)