#!/usr/bin/env bash

TARGET_DIR="data/crello-dataset"
DATASET="crello-dataset-v1"
DOWNLOAD_URL="https://storage.googleapis.com/ailab-public/canvas-vae/${DATASET}.zip"

if [[ ! -d ${TARGET_DIR} ]]; then
    mkdir -p ${TARGET_DIR}
fi

cd ${TARGET_DIR} && { curl -O ${DOWNLOAD_URL}; unzip ${DATASET}; rm ${DATASET}.zip; cd -; }
