import torch
from PIL import Image
import cv2
import sys
import time
import threading
import requests
import base64

# =============================
# 服务器地址
# =============================

SERVER_URL = "http://127.0.0.1:8000/infer"

# =============================
# 状态变量
# =============================

cap = cv2.VideoCapture(0)

running_flag = True
infer_running_flag = True
prompt_editing_flag = False

latest_frame = None

prompt_lock = threading.Lock()
current_prompt = "分析图像内容"

frame_event = threading.Event()

# =============================
# 图片编码
# =============================

def frame_to_base64(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    import io
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")

    return base64.b64encode(buffer.getvalue()).decode()

# =============================
# 推理请求线程
# =============================

def inference_loop():

    global latest_frame

    while running_flag:

        if not infer_running_flag or prompt_editing_flag:
            time.sleep(0.05)
            continue

        frame_event.wait(timeout=0.05)

        if latest_frame is None:
            continue

        frame = latest_frame.copy()
        frame_event.clear()

        try:

            with prompt_lock:
                prompt_snapshot = current_prompt

            img_b64 = frame_to_base64(frame)

            data = {
                "prompt": prompt_snapshot,
                "image_base64": img_b64
            }

            start = time.time()

            response = requests.post(
                SERVER_URL,
                json=data,
                timeout=60
            )

            result = response.json()

            print("\n" + "═"*60)
            print(f"推理耗时: {time.time()-start:.2f}s")
            print(result)
            print("═"*60)

        except Exception as e:
            print("推理错误:", e)


# =============================
# Prompt输入线程
# =============================

def prompt_input_worker():

    global current_prompt
    global prompt_editing_flag

    prompt_editing_flag = True

    text = input("\n输入新prompt (exit退出):\n")

    if text.strip().lower() != "exit":
        with prompt_lock:
            current_prompt = text

        print("✅ Prompt更新成功")

    prompt_editing_flag = False


# =============================
# 启动推理线程
# =============================

threading.Thread(
    target=inference_loop,
    daemon=True
).start()

# =============================
# 主循环
# =============================

print("\n控制说明:")
print("空格 → 修改prompt")
print("s → 暂停推理")
print("r → 恢复推理")
print("q → 退出")

while running_flag:

    ret, frame = cap.read()
    if not ret:
        break

    latest_frame = frame.copy()
    frame_event.set()

    display = frame.copy()

    if prompt_editing_flag:
        cv2.putText(display,
                    "Prompt Editing...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

    if not infer_running_flag:
        cv2.putText(display,
                    "Inference Paused",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    2)

    cv2.imshow("FastVLM Client Camera", display)

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