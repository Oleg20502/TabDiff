#!/bin/bash
set -e

for N in 2 5 10 20 30 40 100; do
    python main.py \
        --gpu 0 \
        --exp_name try_1 \
        --mode test \
        --dataname adult \
        --report \
        --eval_dir N_$N \
        --num_timesteps $N \
        --sample_batch_size 10000
done
