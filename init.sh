#! /usr/bin/bash

pip install -r requirements.txt

git config --global user.email "daniel@d-wetzel.de"
git config --global user.name "Daniel Wetzel"
git config --global credential.helper store

git submodule init
git submodule update