#!/bin/bash

python main.py \
    --gpu 2 --exp_name try_1 \
    --dataname adult --mode train --config tabdiff_configs.toml \
    --num_samples_to_generate 10000
