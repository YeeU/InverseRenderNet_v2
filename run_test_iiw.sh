#! /bin/bash
set -x
ROOT="/shared/storage/cs/staffstore/yy1571"

### input and output options ####
TESTING_MODE="iiw"
MODEL_PATH="iiw_model_ckpt"
IMAGES_DIR="${ROOT}/Data/testData/iiw-dataset/data"
RESULTS_DIR="test_iiw"

python test.py \
    --mode "${TESTING_MODE}" \
    --model "${MODEL_PATH}" \
    --iiw "${IMAGES_DIR}" \
    --output "${RESULTS_DIR}"
