#! /usr/bin/bash

docker run -v $(pwd)/litellm_config.yaml:/app/config.yaml -p 4000:4000 ghcr.io/berriai/litellm:main-latest --num_workers 8 --config /app/config.yaml