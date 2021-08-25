#!/usr/bin/env bash

export PYTHONPATH="src/preprocess"

python -m preprocess crello-document \
    --input-path "data/crello-dataset" \
    --output-path "data/crello-document" \
    --encoder-path "data/pixelvae/encoder" \
    --runner DirectRunner \
    $@
