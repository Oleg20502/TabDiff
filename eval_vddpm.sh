#!/bin/bash

python main.py \
    --gpu 1 \
    --exp_name rec_2_heads_kl_0.1_decay_x0 \
    --config tabdiff_configs_variational_3.toml \
    --mode test \
    --dataname adult \
    --report \
    --eval_dir N_50 \
    --num_timesteps 50 \
    --sample_batch_size 10000
