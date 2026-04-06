#!/bin/bash

python test.py \
    --device cuda:3 \
    --run-dir runs/adult/try_1 \
    --config tabdiff/configs/test/report.yaml
