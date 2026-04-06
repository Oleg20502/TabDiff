#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"

python scripts/debug_mle_fit.py --samples runs/adult/try_1/test/default/0/samples.csv
