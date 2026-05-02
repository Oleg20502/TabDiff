#!/bin/bash

python main.py \
    --gpu 1 \
    --exp_name rec_2_heads_kl_0.1_decay_x0_ld_16 \
    --dataname adult \
    --mode train \
    --config tabdiff_configs_variational_decay_small.toml
