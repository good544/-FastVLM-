import torch
from PIL import Image
import sys
import time

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
print("优化版本推理测试")
print("=" * 70)

print("\n[加载模型...]")
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

t_model = time.time() - t0
print(f"✓ 模型加载完成 ({t_model:.2f}秒)\n")

# 加载图片
image = Image.open(IMAGE_PATH).convert('RGB')
prompt = f"{DEFAULT_IMAGE_TOKEN}\n请简要描述这张图片。"  # 更简短的 prompt

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
image_tensor = process_images([image], image_processor, model.config)[0]
image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

print("=" * 70)
print("方案对比")
print("=" * 70)

# ============ 方案 1: 贪心解码 + 128 tokens ============
print("\n[方案1] 贪心解码 + max_new_tokens=128")
t0 = time.time()

with torch.inference_mode():
    output1 = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        num_beams=1,
        max_new_tokens=128,
        use_cache=True
    )

t1 = time.time() - t0
tokens1 = len(output1[0]) - input_ids.shape[1]
print(f"✓ 耗时: {t1:.2f}秒, 生成: {tokens1} tokens, 速度: {tokens1/t1:.1f} tokens/秒")
response1 = tokenizer.decode(output1[0, input_ids.shape[1]:], skip_special_tokens=True)
print(f"  内容: {response1[:50]}...")

# ============ 方案 2: 采样 + 低 temperature ============
print("\n[方案2] 采样 (temperature=0.3) + max_new_tokens=128")
t0 = time.time()

with torch.inference_mode():
    output2 = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        top_k=50,
        num_beams=1,
        max_new_tokens=128,
        use_cache=True
    )

t2 = time.time() - t0
tokens2 = len(output2[0]) - input_ids.shape[1]
print(f"✓ 耗时: {t2:.2f}秒, 生成: {tokens2} tokens, 速度: {tokens2/t2:.1f} tokens/秒")
response2 = tokenizer.decode(output2[0, input_ids.shape[1]:], skip_special_tokens=True)
print(f"  内容: {response2[:50]}...")

# ============ 方案 3: 最激进优化（64 tokens）============
print("\n[方案3] 贪心解码 + max_new_tokens=64 (超快)")
t0 = time.time()

with torch.inference_mode():
    output3 = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=[image.size],
        do_sample=False,
        num_beams=1,
        max_new_tokens=64,
        use_cache=True
    )

t3 = time.time() - t0
tokens3 = len(output3[0]) - input_ids.shape[1]
print(f"✓ 耗时: {t3:.2f}秒, 生成: {tokens3} tokens, 速度: {tokens3/t3:.1f} tokens/秒")
response3 = tokenizer.decode(output3[0, input_ids.shape[1]:], skip_special_tokens=True)
print(f"  内容: {response3[:50]}...")

print("\n" + "=" * 70)
print("推荐方案: 方案1 (贪心 + 128 tokens)")
print(f"- 总耗时: {t1:.2f}秒 (相比原来 20.84秒 快 {20.84/t1:.1f}倍)")
print(f"- 生成速度: {tokens1/t1:.1f} tokens/秒")
print("=" * 70)
