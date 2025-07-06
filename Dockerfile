FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

WORKDIR /workspace

ENV PYTHONPATH=/workspace
ENV CUDA_VISIBLE_DEVICES=0

RUN apt-get update

RUN pip install --upgrade pip

COPY requirements.txt /workspace/requirements.txt

RUN pip install -r requirements.txt

COPY . /workspace/

RUN mkdir -p /workspace/checkpoints \
    && mkdir -p /workspace/configs \
    && mkdir -p /workspace/models

RUN chmod -R 755 /workspace

EXPOSE 6006

CMD ["bash"]
