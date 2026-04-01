#!/bin/bash

python main.py \
    --gpu 0 \
    --exp_name rec_unimod_kl_0.01 \
    --dataname adult \
    --mode train \
    --config tabdiff_configs_variational.toml
