#! /bin/bash
set -x

### input and output options ####
TESTING_MODE="demo_im"
MODEL_PATH="model_ckpt"
IMAGE_PATH="demo_im.jpg"
MASK_PATH="demo_mask.jpg"
RESULTS_DIR="test_demo"

python test.py \
    --mode "${TESTING_MODE}" \
    --model "${MODEL_PATH}" \
    --image "${IMAGE_PATH}" \
    --mask "${MASK_PATH}" \
    --output "${RESULTS_DIR}"
