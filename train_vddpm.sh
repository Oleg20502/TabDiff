#!/bin/bash

python main.py \
    --gpu 3 \
    --exp_name rec_2_heads_kl_1.0 \
    --dataname adult \
    --mode train \
    --config tabdiff_configs_variational_2.toml
