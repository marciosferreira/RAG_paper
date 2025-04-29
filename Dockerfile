FROM jupyter/base-notebook

# Set environment variables
ENV JUPYTER_PORT=8888 \
    OLLAMA_PORT=11434

USER root

# Install required tools.
RUN apt-get -qq update \
    && apt-get -qq --no-install-recommends install apt-utils vim-tiny \
    tcpdump nano wget sudo iputils-ping s3fs htop curl git ffmpeg libsm6 \
    openssh-client libxext6 build-essential python3-dev python3-pip python3-venv \
    && apt-get -qq clean    \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /setup/requirements.txt
COPY ./setup* /setup/
RUN chmod a+x /setup/setup*

# Expose necessary ports
EXPOSE $JUPYTER_PORT $OLLAMA_PORT

RUN echo "jovyan ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
USER jovyan

RUN pip install --no-cache-dir --upgrade pip wheel setuptools \
    && pip install --no-cache-dir -r /setup/requirements.txt \
    && python -m pip cache purge
RUN pip uninstall setuptools -y && pip install --no-cache-dir docling