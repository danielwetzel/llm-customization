#! /usr/bin/bash

python3 -m conda activate pytorch
python3 -m pip install -r requirements.txt

git config --global user.email "daniel@d-wetzel.de"
git config --global user.name "Daniel Wetzel"

