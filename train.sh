#!/bin/bash

python train.py \
    --device cuda:2 \
    --config tabdiff/configs/train/unimod_mlp/base.yaml \
    --run-name try_1
