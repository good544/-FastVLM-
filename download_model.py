# from huggingface_hub import snapshot_download
# import os

# # 目标文件夹（会自动创建子文件夹 FastVLM-0.5B）
# local_dir = r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B"

# print(f"开始下载 apple/FastVLM-0.5B 到 {local_dir} ...")
# print("模型约 1.5-2GB，第一次可能较慢，支持断点续传")

# # 下载整个仓库（包括 config、tokenizer、safetensors 等所有文件）
# snapshot_download(
#     repo_id="apple/FastVLM-0.5B",
#     local_dir=local_dir,
#     local_dir_use_symlinks=False,  # Windows 上推荐 False，避免符号链接问题
#     resume_download=True           # 支持断点续传
# )

# print("下载完成！检查文件夹是否完整。")

# 新建文件 download_fp16.py 并运行
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="apple/FastVLM-0.5B",
    local_dir=r"D:\fastvlm\ml-fastvlm\models\FastVLM-0.5B",
    local_dir_use_symlinks=False,      # Windows 必须
    resume_download=True               # 断点续传
)