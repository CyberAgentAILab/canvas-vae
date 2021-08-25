#!/usr/bin/env bash

export PYTHONPATH="src/preprocess"

RICO_URL="gs://crowdstf-rico-uiuc-4540/rico_dataset_v0.1/semantic_annotations.zip"

python -m preprocess rico \
    --input-path ${RICO_URL} \
    --output-path "data/rico" \
    --runner DirectRunner \
    $@
