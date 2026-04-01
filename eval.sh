#!/bin/bash

python main.py \
    --gpu 3 \
    --exp_name try_1 \
    --config tabdiff_configs_variational.toml \
    --ckpt_path tabdiff/ckpt/adult/try_1_variational/model_350.pt \
    --mode test \
    --dataname adult \
    --num_samples_to_generate 10000 \
    --report