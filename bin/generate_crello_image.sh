#!/usr/bin/env bash

export PYTHONPATH="src/preprocess"

python -m preprocess crello-image \
    --input-path "data/crello-dataset" \
    --output-path "data/crello-image" \
    --runner DirectRunner \
    $@
