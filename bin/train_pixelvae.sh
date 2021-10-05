#!/usr/bin/env bash

export PYTHONPATH="src/pixel-vae"

NOW=$(date '+%Y%m%d%H%M%S')

python -m pixelvae.main \
    --data-dir "data/crello-image" \
    --job-dir "tmp/jobs/pixelvae/${NOW}" \
    --output-path "data/pixelvae/encoder" \
    $@
