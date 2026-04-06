#!/bin/bash

python test.py \
    --device cuda:2 \
    --run-dir runs/adult/rec_2_heads_kl_0_1 \
    --config tabdiff/configs/test/report.yaml
