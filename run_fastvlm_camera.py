import torch
from PIL import Image
import cv2
import sys
import time
import os

# 添加本地路径，让 Python 找到 llava_qwen.py 和 llava.mm_utils
sys.path.append(r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B")

from transformers import AutoTokenizer
from llava_qwen import LlavaQwen2ForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images  # 用于手动 token 处理
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

# ==================== 配置区 ====================
MODEL_PATH = r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

USE_8BIT = False

print("正在加载 FastVLM-0.5B ...")
print(f"设备: {DEVICE}   数据类型: {DTYPE}")
print(f"模型路径: {MODEL_PATH}\n")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

loading_kwargs = {
    "torch_dtype": DTYPE,
    "device_map": "auto" if DEVICE == "cuda" else None,
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
}

if USE_8BIT and DEVICE == "cuda":
    from transformers import BitsAndBytesConfig
    loading_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

model = LlavaQwen2ForCausalLM.from_pretrained(MODEL_PATH, **loading_kwargs)
model.eval()

# 获取 image_processor
image_processor = model.get_vision_tower().image_processor

print("模型加载完成！\n")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("摄像头打开失败，尝试改成 cv2.VideoCapture(1)")
    sys.exit(1)

print("操作说明：空格推理，c改prompt，q退出\n")

# 默认 prompt，必须包含 <image>
IMAGE_TOKEN = "<image>"
current_prompt = f"{IMAGE_TOKEN}\n请用中文详细描述这张图片的内容，包括场景、主要物体、颜色、人物动作、氛围等。"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera - 空格推理｜c改prompt｜q退出", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        print("\n请输入新问题（**必须包含 <image>**）：")
        lines = []
        while True:
            line = input()
            if not line.strip():
                break
            lines.append(line)
        new_prompt = " ".join(lines).strip()
        if new_prompt:
            if IMAGE_TOKEN not in new_prompt:
                print(f"警告：缺少 {IMAGE_TOKEN}，图像可能无法处理！")
            current_prompt = new_prompt
        print(f"当前prompt：{current_prompt}\n")

    elif key == 32:
        print("\n正在推理...")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        try:
            prompt = current_prompt

            # 使用 tokenizer_image_token 处理包含 <image> 的提示
            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
            
            # 使用 process_images 处理图像，并转换为 float16
            image_tensor = process_images([pil_image], image_processor, model.config)[0]
            image_tensor = image_tensor.unsqueeze(0).half().to(model.device)

            print("input_ids shape:", input_ids.shape)
            print("image_tensor shape:", image_tensor.shape)

            start_time = time.time()

            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[pil_image.size],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=64,  # 优化：从 256 减少到 64
                    use_cache=True
                )

            input_len = input_ids.shape[1]
            answer_ids = generated_ids[0, input_len:]
            response = tokenizer.decode(answer_ids, skip_special_tokens=True)

            end_time = time.time()

            print("\n" + "═" * 70)
            print(f"Time: {end_time - start_time:.2f} seconds")
            print("Response:")
            print(response.strip())
            print("═" * 70 + "\n")

        except Exception as e:
            print("推理失败：", str(e))
            import traceback
            traceback.print_exc()

cap.release()
cv2.destroyAllWindows()
print("程序已退出")