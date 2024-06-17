#! /usr/bin/bash

python3 -m conda init
python3 -m conda activate pytorch
python3 -m pip install -r requirements.txt

git config --global user.email "daniel@d-wetzel.de"
git config --global user.name "Daniel Wetzel"
git config --global credential.helper store

git submodule init
git submodule update

conda create -n llm_judge python=3.9 -y
conda init

exec bash