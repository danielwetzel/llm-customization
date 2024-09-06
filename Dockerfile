# Dockerfile

FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Python script and requirements
COPY streamlit-visualization.py requirements.txt ./

# Copy the directories results and pages
COPY results/ ./results/
COPY pages/ ./pages/
COPY .streamlit/ ./.streamlit/
COPY llm_judge/results/ ./llm_judge/results/

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit-visualization.py", "--server.port=8501", "--server.address=0.0.0.0"]