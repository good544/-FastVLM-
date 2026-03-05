import torch
import sys
import base64
import io
import time

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from PIL import Image

# =============================
# ⭐ 项目根路径（最重要）
# =============================

PROJECT_ROOT = r"D:\fastvlm\ml-fastvlm"
MODEL_PATH = r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B"

sys.path.append(PROJECT_ROOT)

# =============================
# 导入 FastVLM 模型
# =============================

from llava_qwen import LlavaQwen2ForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX

# =============================
# FastAPI
# =============================

app = FastAPI(title="FastVLM Server")

# =============================
# 设备
# =============================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("🚀 加载模型...")

# =============================
# Tokenizer
# =============================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

# =============================
# 模型加载（只加载一次）
# =============================

model = LlavaQwen2ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

model.eval()

image_processor = model.get_vision_tower().image_processor

print("✅ 模型加载完成")

# =============================
# 请求结构
# =============================

class InferRequest(BaseModel):
    prompt: str
    image_base64: str

# =============================
# base64解码
# =============================

def decode_image(b64_string):
    if "base64," in b64_string:
        b64_string = b64_string.split("base64,")[1]

    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

# =============================
# 推理接口
# =============================

@app.post("/infer")
async def infer(req: InferRequest):

    try:
        start = time.time()

        image = decode_image(req.image_base64)

        prompt = f"<image>\n{req.prompt}"

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(model.device)

        image_tensor = process_images(
            [image],
            image_processor,
            model.config
        )[0].unsqueeze(0).to(model.device)

        if DEVICE == "cuda":
            image_tensor = image_tensor.half()

        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                num_beams=1,
                max_new_tokens=64,
                use_cache=True
            )

        input_len = input_ids.shape[1]

        response = tokenizer.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

        return {
            "result": response.strip(),
            "time": round(time.time() - start, 2)
        }

    except Exception as e:
        return {"error": str(e)}

# =============================
# 服务器启动
# =============================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )

    # uvicorn server:app --host 127.0.0.1 --port 8000