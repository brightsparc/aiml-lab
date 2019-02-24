# Build an image that can do training and inference in SageMaker
# This is a Python 2 image that uses the nginx, gunicorn, flask stack
# for serving inferences in a stable way.

FROM ubuntu:16.04

LABEL maintainer="Julian Bright <brightsparc@gmail.com>"

RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    libopencv-dev \
    python3-dev \
    build-essential \
    ca-certificates \
    wget \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Symlink /usr/bin/python to the python version we're building for.
RUN rm /usr/bin/python && ln -s "/usr/bin/python3" /usr/bin/python

# Here we get all python packages.
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py && \
    pip install mxnet opencv-python flask gevent gunicorn && \
        rm -rf /root/.cache

# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY app /opt/program
WORKDIR /opt/program

# Default the entrypoint to serve
ENTRYPOINT ["serve"]