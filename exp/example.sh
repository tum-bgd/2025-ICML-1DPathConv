#!/bin/bash
# operations for apptainer

python -m pip install networkx numpy matplotlib imageio timm cupy-cuda12x PuLP scikit-learn ninja
python ./src/main.py --epoch 300 --bs 384 --dataset in64 --path g --model s --opt adamw --initlr 0.01
