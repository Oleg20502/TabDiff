#!/bin/bash

python main.py \
    --gpu 3 \
    --exp_name try_1 \
    --mode test \
    --dataname adult \
    --report \
    --eval_dir N_50 \
    --num_timesteps 50