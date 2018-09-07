#!/usr/bin/env bash

python3 svm-FB_train.py
python3 svm-TW_train.py
python3 svm-FB-TW.py
python3 svm-TW-FB.py

cd ../CNN
bash run_CNN.sh