FROM python:3.13-slim

WORKDIR /work

COPY requirements/robustness.txt /tmp/robustness.txt
RUN pip install --no-cache-dir -r /tmp/robustness.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
