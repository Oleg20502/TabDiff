#!/bin/bash

python main.py \
    --gpu 0 --exp_name tabdiff \
    --dataname adult --mode train --config tabdiff_configs.toml
