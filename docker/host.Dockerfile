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

# The host image is the flow *driver* (metaflow + docker CLI), not a pipeline
# step, so a single image launches any flow. Pass the flow file as the command:
#   docker run pao-host:dev flows/flow.py run
#   docker run pao-host:dev flows/monitor_flow.py run --model_id <id>
#   docker run pao-host:dev flows/ab_flow.py run --run_id_a <a> --run_id_b <b>
ENTRYPOINT ["python"]
CMD ["flows/flow.py", "run"]
