#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source ${SCRIPT_DIR}/env.sh
source ${SCRIPT_DIR}/ape_utils.sh

SRC_FILE="$DEV.bpe.$SRC"
SRC_PATH="${WMT_PATH}/${SRC_FILE}"
MT_PATH="${WMT_PATH}/$DEV.$MT"

GPU_ID=$1
MODEL_PATH=$2

PRED_DIR="$(dirname ${SRC_PATH})/pred"
MODEL_FILE="$(basename ${MODEL_PATH})"
PSEUDO_REF_PATH="${PRED_DIR}/${SRC_FILE}_${MODEL_FILE}"

# Create Prediction directory if it does not exist
[[ -d ${PRED_DIR} ]] || mkdir ${PRED_DIR}

# Predict if gpu id is given
if [[ ! -e ${PSEUDO_REF_PATH} ]]; then
    translate ${MODEL_PATH} ${SRC_PATH} ${PSEUDO_REF_PATH} ${GPU_ID}
fi

# Postprocess Data
postprocess ${PSEUDO_REF_PATH}

# Score
tercom ${PSEUDO_REF_PATH} ${MT_PATH}


