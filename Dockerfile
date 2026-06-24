# base image with cuda 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# install python 3.11, pip, and opencv dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# set python3.11 as the default python
RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

# install uv
RUN pip install uv

# create venv
ENV PATH="/.venv/bin:${PATH}"
RUN uv venv --python 3.11 /.venv

# install dependencies
RUN uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu121

# install remaining dependencies from PyPI
COPY requirements.txt /requirements.txt
RUN uv pip install -r /requirements.txt

# install IP-Adapter FaceID dependencies
RUN uv pip install insightface onnxruntime-gpu opencv-python-headless

# copy files
COPY download_weights.py schemas.py handler.py test_input.json /

# Download ALL model weights during build — bakes them into the image
# so workers start instantly without re-downloading on every cold start.
RUN python /download_weights.py

# run the handler
CMD python -u /handler.py
