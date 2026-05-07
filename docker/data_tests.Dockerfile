FROM python:3.13-slim

WORKDIR /work

COPY requirements/data_tests.txt /tmp/data_tests.txt
RUN pip install --no-cache-dir -r /tmp/data_tests.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
