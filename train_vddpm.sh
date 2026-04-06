#!/bin/bash

python train.py \
    --device cuda:3 \
    --config tabdiff/configs/train/unimod_mlp/variational.yaml \
    --run-name rec_2_heads_kl_1_0
