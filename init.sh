#! /usr/bin/bash

pip install -r dev_requirements.txt

git config --local user.email "test@d-wetzel.de"
git config --local user.name "Thesis Reviewer"
git config --local credential.helper store

git submodule init
git submodule update