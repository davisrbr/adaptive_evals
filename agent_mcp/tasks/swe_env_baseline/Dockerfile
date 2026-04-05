FROM python:3.12-slim

WORKDIR /testbed

# Copy all testbed files
COPY testbed/ /testbed/

# Create directory structure for fileutil test
RUN mkdir -p /testbed/configs/sub && \
    echo 'key: value' > /testbed/configs/main.yaml && \
    echo 'nested: true' > /testbed/configs/sub/extra.yaml

CMD ["tail", "-f", "/dev/null"]
