import torch
from PIL import Image
import sys
import time

# 添加本地路径
sys.path.append(r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B-fp16")

from transformers import AutoTokenizer
from llava_qwen import LlavaQwen2ForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# ==================== 配置 ====================
MODEL_PATH = r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B"  # 改为普通版本而非 fp16
IMAGE_PATH = r"D:\fastvlm\ml-fastvlm\testpicture2.png"  # 改成你的图片路径

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16  # 用 float16

print("正在加载 FastVLM-0.5B ...")
print(f"设备: {DEVICE}   数据类型: {DTYPE}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

loading_kwargs = {
    "torch_dtype": DTYPE,
    "device_map": "auto" if DEVICE == "cuda" else None,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}

model = LlavaQwen2ForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)
model.eval()

# 获取 image_processor
image_processor = model.get_vision_tower().image_processor

print("模型加载完成！\n")

# 加载图片
try:
    image = Image.open(IMAGE_PATH).convert('RGB')
    print(f"图片加载成功: {IMAGE_PATH}")
    print(f"图片大小: {image.size}\n")
except FileNotFoundError:
    print(f"错误：找不到图片 {IMAGE_PATH}")
    sys.exit(1)

# 准备 prompt
prompt = f"{DEFAULT_IMAGE_TOKEN}\n请用中文详细描述这张图片的内容，包括场景、主要物体、颜色、人物动作、氛围等。"

print(f"Prompt: {prompt}\n")

# 处理输入
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

# 处理图像
image_tensor = process_images([image], image_processor, model.config)[0]
image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

print("input_ids shape:", input_ids.shape)
print("image_tensor shape:", image_tensor.shape)
print("\n正在进行推理...\n")

start_time = time.time()

# 执行推理（对标官方 predict.py）
with torch.inference_mode():
    generated_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,  # 不采样
        num_beams=1,
        max_new_tokens=256,
        use_cache=True
    )

input_len = input_ids.shape[1]
answer_ids = generated_ids[0, input_len:]

print(f"Debug - 生成的总 tokens: {generated_ids.shape[1]}")
print(f"Debug - 输入 tokens: {input_len}")
print(f"Debug - 输出 tokens: {len(answer_ids)}")
print(f"Debug - 输出 token IDs: {answer_ids.tolist()}")

if len(answer_ids) > 0:
    response = tokenizer.decode(answer_ids, skip_special_tokens=True)
else:
    response = "[空白回答]"

end_time = time.time()

print("=" * 70)
print(f"耗时: {end_time - start_time:.2f} 秒")
print("回答：")
print(response.strip())
print("=" * 70)
