FROM python:3.13-slim

WORKDIR /work

COPY docker/requirements/robustness.txt /tmp/robustness.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    -r /tmp/robustness.txt

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
