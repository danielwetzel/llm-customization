#! /usr/bin/bash

conda deactivate
python3 -m conda activate llm_judge

cd llm_judge/FastChat
pip install -e ".[model_worker,llm_judge]"



