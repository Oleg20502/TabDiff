#!/bin/bash

for N in 1 5 10 20 30 40 50; do
    python test.py \
        --device cuda:0 \
        --run-dir runs/adult/rec_2_heads_big_kl_0_1 \
        --config tabdiff/configs/test/report.yaml
done
