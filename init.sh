#! /usr/bin/bash

pip install -r dev_requirements.txt

git config --global user.email "test@d-wetzel.de"
git config --global user.name "Thesis Reviewer"
git config --global credential.helper store

git submodule init
git submodule update