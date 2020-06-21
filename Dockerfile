FROM nvcr.io/nvidia/pytorch:20.03-py3

ENV NVIDIA_VISIBLE_DEVICES all

RUN DEBIAN_FRONTEND=noninteractive apt-get update
WORKDIR /workspace
COPY ./project/ /workspace
RUN pip install -r requirements.txt
