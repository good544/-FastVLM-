import torch
from PIL import Image
import sys
import time
import numpy as np

# 添加本地路径
sys.path.append(r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B")

from transformers import AutoTokenizer
from llava_qwen import LlavaQwen2ForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# ==================== 配置 ====================
MODEL_PATH = r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B"
IMAGE_PATH = r"D:\fastvlm\ml-fastvlm\testpicture2.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

print("=" * 70)
print("性能分析")
print("=" * 70)

# ============ 1. 模型加载耗时 ============
print("\n[1] 模型加载...")
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
loading_kwargs = {
    "torch_dtype": DTYPE,
    "device_map": "auto" if DEVICE == "cuda" else None,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}
model = LlavaQwen2ForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)
model.eval()
image_processor = model.get_vision_tower().image_processor

t1 = time.time()
print(f"✓ 模型加载耗时: {t1-t0:.2f} 秒")

# ============ 2. 图片加载耗时 ============
print("\n[2] 图片加载...")
t0 = time.time()

image = Image.open(IMAGE_PATH).convert('RGB')

t1 = time.time()
print(f"✓ 图片加载耗时: {t1-t0:.2f} 秒")

# ============ 3. Tokenize耗时 ============
print("\n[3] Tokenize...")
prompt = f"{DEFAULT_IMAGE_TOKEN}\n请用中文详细描述这张图片。"

t0 = time.time()
input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
t1 = time.time()
print(f"✓ Tokenize耗时: {t1-t0:.3f} 秒")

# ============ 4. 图片处理耗时 ============
print("\n[4] 图片处理...")
t0 = time.time()

image_tensor = process_images([image], image_processor, model.config)[0]
image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

t1 = time.time()
print(f"✓ 图片处理耗时: {t1-t0:.3f} 秒")

# ============ 5. 预热（第一次推理较慢） ============
print("\n[5] CUDA 预热（第一次推理）...")
t0 = time.time()

with torch.inference_mode():
    _ = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        num_beams=1,
        max_new_tokens=10,
        use_cache=True
    )

t1 = time.time()
print(f"✓ 预热耗时: {t1-t0:.2f} 秒")

# ============ 6. 实际推理耗时 ============
print("\n[6] 实际推理（max_new_tokens=256）...")
t0 = time.time()

with torch.inference_mode():
    generated_ids = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        num_beams=1,
        max_new_tokens=256,
        use_cache=True
    )

t1 = time.time()
total_gen_time = t1 - t0
print(f"✓ 推理耗时: {total_gen_time:.2f} 秒")

# ============ 7. 解码耗时 ============
print("\n[7] 解码...")
t0 = time.time()

input_len = input_ids.shape[1]
answer_ids = generated_ids[0, input_len:]
response = tokenizer.decode(answer_ids, skip_special_tokens=True)

t1 = time.time()
print(f"✓ 解码耗时: {t1-t0:.3f} 秒")

print("\n" + "=" * 70)
print("生成统计:")
print(f"  - 输入 tokens: {input_len}")
print(f"  - 生成 tokens: {len(answer_ids)}")
print(f"  - 生成速度: {len(answer_ids) / total_gen_time:.1f} tokens/秒")
print(f"  - 生成内容: {response[:50]}...")
print("=" * 70)

# ============ 8. 优化建议测试 ============
print("\n[8] 测试优化参数：减少生成长度（max_new_tokens=128）...")
t0 = time.time()

with torch.inference_mode():
    generated_ids_opt = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        num_beams=1,
        max_new_tokens=128,  # 减半
        use_cache=True
    )

t1 = time.time()
opt_gen_time = t1 - t0
print(f"✓ 优化后耗时: {opt_gen_time:.2f} 秒")
print(f"  - 生成速度: {(len(generated_ids_opt[0]) - input_len) / opt_gen_time:.1f} tokens/秒")
