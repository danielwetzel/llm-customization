#! /usr/bin/bash
docker run -p 9099:9099 --add-host=host.docker.internal:host-gateway -v $(pwd)/pipelines:/app/pipelines ghcr.io/open-webui/pipelines:main
docker run -v $(pwd)/open-webui:/app/backend/data --network="host" -e OPENAI_API_BASE_URLS="http://localhost:8000/v1" -e OPENAI_API_KEYS="EMPTY" ghcr.io/open-webui/open-webui:main
# docker run -v $(pwd)/open-webui:/app/backend/data --network="host" -e OPENAI_API_BASE_URLS="http://localhost:4000/v1;http://localhost:8000/v1" -e OPENAI_API_KEYS="EMPTY;EMPTY" ghcr.io/open-webui/open-webui:main 