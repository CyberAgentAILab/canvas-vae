#!/usr/bin/env bash

export PYTHONPATH="src/canvas-vae"

DATASET=${1:-"crello-document"}
NOW=$(date '+%Y%m%d%H%M%S')

python -m canvasvae \
    --dataset-name ${DATASET} \
    --data-dir "data/${DATASET}" \
    --job-dir "tmp/jobs/canvasvae/${NOW}" \
    ${@:2}
