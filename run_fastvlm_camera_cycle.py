import torch
from PIL import Image
import cv2
import sys
import time
import threading

# ===== 模型路径 =====
sys.path.append(r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B")

from transformers import AutoTokenizer
from llava_qwen import LlavaQwen2ForCausalLM
from llava.mm_utils import tokenizer_image_token, process_images
from llava.constants import IMAGE_TOKEN_INDEX

# ============================
# 配置
# ============================

MODEL_PATH = r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

print("加载模型...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = LlavaQwen2ForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.eval()

image_processor = model.get_vision_tower().image_processor

print("模型加载完成")

# ============================
# 状态变量（核心）
# ============================

cap = cv2.VideoCapture(0)

IMAGE_TOKEN = "<image>"

prompt_lock = threading.Lock()

current_prompt = f"{IMAGE_TOKEN}\n分析特征"

running_flag = True
infer_running_flag = True
prompt_editing_flag = False

latest_frame = None

frame_event = threading.Event()

# ⭐ 推理中断信号（关键）
stop_inference_event = threading.Event()


# ============================
# 推理循环线程（核心🔥）
# ============================

def inference_loop():

    global latest_frame

    while running_flag:

        # 推理暂停
        if not infer_running_flag:
            time.sleep(0.02)
            continue

        # prompt编辑模式暂停推理
        if prompt_editing_flag:
            time.sleep(0.02)
            continue

        frame_event.wait(timeout=0.02)

        if latest_frame is None:
            continue

        frame = latest_frame.copy()
        frame_event.clear()

        # ⭐ 支持中途打断推理
        if stop_inference_event.is_set():
            stop_inference_event.clear()
            continue

        try:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)

            with prompt_lock:
                prompt_snapshot = current_prompt

            input_ids = tokenizer_image_token(
                prompt_snapshot,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors='pt'
            ).unsqueeze(0).to(model.device)

            image_tensor = process_images(
                [pil_image],
                image_processor,
                model.config
            )[0].unsqueeze(0).half().to(model.device)

            if stop_inference_event.is_set():
                stop_inference_event.clear()
                continue

            start = time.time()

            with torch.inference_mode():
                generated_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    image_sizes=[pil_image.size],
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=64,
                    use_cache=True
                )

            if prompt_editing_flag:
                continue

            input_len = input_ids.shape[1]

            response = tokenizer.decode(
                generated_ids[0, input_len:],
                skip_special_tokens=True
            )

            print("\n" + "═"*60)
            print(f"推理耗时: {time.time()-start:.2f}s")
            print(response.strip())
            print("═"*60)

        except Exception as e:
            print("推理错误:", e)


# 启动推理线程
threading.Thread(target=inference_loop, daemon=True).start()


# ============================
# prompt输入线程
# ============================

def prompt_input_worker():

    global current_prompt
    global prompt_editing_flag

    prompt_editing_flag = True

    text = input("\n输入新prompt (exit退出编辑):\n")

    if text.strip().lower() != "exit":
        with prompt_lock:
            current_prompt = f"{IMAGE_TOKEN}\n{text}"

    prompt_editing_flag = False

    # ⭐ 修改 prompt 后立刻允许推理
    print("Prompt更新完成，推理恢复")


# ============================
# 主循环（摄像头显示）
# ============================

print("\n控制说明:")
print("空格 → 修改prompt")
print("s → 暂停推理")
print("r → 恢复推理")
print("q → 退出程序")

while running_flag:

    ret, frame = cap.read()
    if not ret:
        break

    latest_frame = frame.copy()
    frame_event.set()

    display_frame = frame.copy()

    if prompt_editing_flag:
        cv2.putText(
            display_frame,
            "Prompt Editing...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    if not infer_running_flag:
        cv2.putText(
            display_frame,
            "Inference Paused",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

    cv2.imshow("FastVLM Camera", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        running_flag = False
        break

    elif key == ord('s'):
        infer_running_flag = False
        print("推理暂停")

    elif key == ord('r'):
        infer_running_flag = True
        print("推理恢复")

    elif key == 32:  # 空格
        threading.Thread(
            target=prompt_input_worker,
            daemon=True
        ).start()

cap.release()
cv2.destroyAllWindows()