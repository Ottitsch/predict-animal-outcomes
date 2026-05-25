FROM python:3.13-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY docker-cli-bin /usr/bin/docker
RUN chmod +x /usr/bin/docker

WORKDIR /work

COPY docker/requirements/host.txt /tmp/host.txt
RUN pip install --no-cache-dir \
    --trusted-host pypi.org \
    --trusted-host files.pythonhosted.org \
    --trusted-host pypi.python.org \
    -r /tmp/host.txt

ENV PYTHONUNBUFFERED=1
ENV METAFLOW_USER=pipeline

ENTRYPOINT ["python", "monitor_flow.py"]
CMD ["run"]
