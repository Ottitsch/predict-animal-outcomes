FROM python:3.13-slim

WORKDIR /work

COPY docker/requirements/monitor.txt /tmp/monitor.txt
RUN pip install --no-cache-dir \
    -r /tmp/monitor.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
