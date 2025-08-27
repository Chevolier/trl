import os 
#openai api configuration

# Qwen 配置
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_IMAGE_TOKEN = "<image>"
IMAGE_FACTOR = 28
FRAME_FACTOR = 2
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_TOTAL_PIXELS = int(float(os.environ.get('VIDEO_MAX_PIXELS', 128000 * 28 * 28 * 0.9)))
VIDEO_MIN_PIXELS = 128 * 28 * 28   # 降低最小分辨率
VIDEO_MAX_PIXELS =  768 * 28 * 28      # 匹配训练时的video_max_pixels
NFRAMES = 64                  # 匹配训练时的video_maxlen

# VIDEO_MIN_PIXELS = 128 * 28 * 28
# VIDEO_MAX_PIXELS = 768 * 28 * 28  #262144
# NFRAMES = 16


SYSTEM_PROMPT = """
You are an expert video analyst with advanced computer vision capabilities. 
Your task is to carefully analyze frames from the video and provide detailed, accurate descriptions.
You should consider frames as one video

Instructions:
- Analyze all visual elements including people, objects, actions, settings, and characteristics
- Pay attention to specific details like clothing types, facial expressions, body movements, and environmental context
- Provide clear, structured responses based on what you observe
- When asked for lists, format them as requested (e.g., [item1, item2, item3])
- Be precise and factual in your descriptions
- Note any temporal changes or sequences of events
- Consider lighting conditions, time of day, and spatial relationships
- Use specific, concrete language - avoid vague terms like "interaction", "activity", "something", "doing", or "performing"

Always base your analysis solely on what is visible in the video frames.
"""

USER_PROMPT_TEMPLATE = """
Please analyze the video and provide a detailed response in standard JSON format. Your response should include:

1. description: A brief description of the overall situation in the video
2. actions: A list of main actions in format ["subject + action", "subject + action", ...]
3. objects: A list of main objects visible in the video
4. characteristics: A list of characteristics in format ["subject + characteristic", "adjective + subject", ...]
5. title: A short title following the specified guidelines

IMPORTANT: Respond ONLY with valid JSON format. Do not include any additional text or explanations outside the JSON structure.

Example response format:
{
  "description": "A man is walking in a park with his dog",
  "actions": ["a man is walking", "a dog is moving"],
  "objects": ["person", "dog", "trees", "path"],
  "characteristics": ["a man is wearing blue shirt", "a brown dog", "green trees"],
  "title": "man walking dog"
}
"""
