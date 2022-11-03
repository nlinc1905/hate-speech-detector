#!/bin/bash

# run bolt in the background
python src/app.py &

# run dash in the foreground
gunicorn -b 0.0.0.0:8050 --reload src.dashboard:server
