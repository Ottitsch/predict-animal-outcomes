FROM python:3.13-slim

WORKDIR /work

COPY docker/requirements/ab.txt /tmp/ab.txt
RUN pip install --no-cache-dir \
    -r /tmp/ab.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
