#!/usr/bin/env bash

export PYTHONPATH="src/preprocess"

RICO_PATH=${1:-"tmp/rico_dataset_v0.1_semantic_annotations.zip"}

python -m preprocess rico \
    --input-path ${RICO_PATH} \
    --output-path "data/rico" \
    --runner DirectRunner \
    --direct_num_workers 0 \
    --direct_running_mode "multi_threading" \
    $@
