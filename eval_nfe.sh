#!/bin/bash

for N in 1 5 10 20 30 40 50; do
    python main.py \
        --gpu 0 \
        --exp_name rec_2_heads_big_kl_0.1 \
        --config tabdiff_configs_variational_3.toml \
        --mode test \
        --dataname adult \
        --report \
        --eval_dir N_$N \
        --num_timesteps $N \
        --sample_batch_size 10000
done