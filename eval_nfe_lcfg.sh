#!/bin/bash
set -e

for N in 4 5 7 10 14 20 30 40 50 100; do
    python main.py \
        --gpu 0 \
        --exp_name rec_2_heads_big_kl_0.01_variational \
        --config tabdiff_configs_variational_3.toml \
        --mode test \
        --dataname adult \
        --report \
        --eval_dir N_${N} \
        --num_timesteps $N \
        --sample_batch_size 10000
done
