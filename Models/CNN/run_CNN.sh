#!/usr/bin/env bash

echo 'Running CNN'
python3 cnn_FB.py
python3 cnn_TW.py
python3 cnn_FB-TW.py
python3 cnn_TW-FB.py
