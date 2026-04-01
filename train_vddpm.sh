#!/bin/bash

python main.py \
    --gpu 0 --exp_name try_1 \
    --dataname adult --mode train --config tabdiff_configs_variational.toml \
    --num_samples_to_generate 10000
