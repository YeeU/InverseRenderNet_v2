#! /bin/bash
set -x
ROOT="/shared/storage/cs/staffstore/yy1571"

### input and output options ####
TESTING_MODE="diode"
MODEL_PATH="diode_model_ckpt"
IMAGES_DIR="${ROOT}/Data/DIODE"
RESULTS_DIR="test_diode"

python test.py \
    --mode "${TESTING_MODE}" \
    --model "${MODEL_PATH}" \
    --diode "${IMAGES_DIR}" \
    --output "${RESULTS_DIR}"
