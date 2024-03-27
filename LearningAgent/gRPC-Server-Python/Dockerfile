FROM sadraiiali/rcss2d-grpc-agent:latest

ENV TEAM_NAME=team
ENV HOST=127.0.0.1
ENV PORT=6000

RUN apt-get clean && apt-get update --allow-insecure-repositories && \
    DEBIAN_FRONTEND="noninteractive" apt-get -y install \
    python3 \
    python3-pip \
    python3-dev \
    python3-setuptools \
    python3-wheel \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/code/requirements.txt
RUN pip3 install -r /app/code/requirements.txt --break-system-packages

COPY . /app/code

CMD ["bash", "/app/code/docker-entrypoint.sh"]