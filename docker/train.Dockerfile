FROM python:3.13-slim

WORKDIR /work

COPY requirements/train.txt /tmp/train.txt
RUN pip install --no-cache-dir -r /tmp/train.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
