# Your image MUST be built on the ROCm base image AMD mandates. This is checked
# by layer identity before your container is run -- we compare the lower layers
# of your image against the base itself, not the name you gave it -- and an
# image that is not built on it is disqualified.
#
# It is the same image your development session runs, and it supports both
# gfx942 (MI300X, CDNA) and gfx1100 (Navi 31, RDNA3), so one submission runs on
# whichever GPU you are given. It ships torch and PIL ONLY; add whatever else
# you need to requirements.txt.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

# Parsers, FAISS, and OCR do not depend on torch.
# transformers does. Installed with --no-deps so pip cannot replace the ROCm
# torch in this image with a CUDA build. The assert fails the build if it did.
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps transformers \
 && pip install --no-cache-dir filelock huggingface-hub packaging pyyaml regex requests safetensors tokenizers tqdm \
 && python3 -c "import torch; v=torch.__version__; assert '+rocm' in v, v; print('torch', v)"

COPY app/ /app/

# Weights are downloaded while the build still has a network. The running
# container does not. HF_HUB_OFFLINE is set only after the download.
ENV HF_HOME=/models/hf \
    FASTEMBED_CACHE_PATH=/models/fastembed \
    MC3_LLM=Qwen/Qwen2.5-1.5B-Instruct \
    MC3_SOCKET=/tmp/mc3-answer.sock \
    MC3_WAIT_SOCKET=20 \
    PYTHONUNBUFFERED=1

RUN mkdir -p /models/hf /models/fastembed /app/corpus /app/output /app/index \
 && python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B-Instruct')" \
 && python3 -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/models/fastembed')"

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The server loads the model once and stays up. Each question is an exec of
# app.py, which only sends the question over the socket.
CMD ["python3", "/app/serve.py"]
