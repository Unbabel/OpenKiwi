#!/bin/bash

SRC_PATH=$1
GOLD_PATH=$2
MODEL_PATH=$3
GPU_ID=$4

PRED_DIR=$(dirname ${SRC_PATH})
MODEL_FILE=$(basename ${MODEL_PATH})
PRED_PATH=${PRED_DIR}/${SRC_FILE}_${MODEL_FILE}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source $SCRIPT_DIR/ape_utils.sh

# Create Prediction directory if it does not exist
[[ -d ${PRED_DIR} ]] || mkdir ${PRED_DIR}

# Predict if gpu id is given
if [[ "${GPU_ID}" != "" ]]; then
    translate ${MODEL_PATH} ${SRC_PATH} ${PRED_PATH} ${GPU_ID}
fi


# Postprocess Data
remove_bpe ${PRED_PATH}
remove_bpe ${GOLD_PATH}

postprocess ${PRED_PATH}
# Score
tercom ${REF_PATH} ${HYP_PATH}
echo $HYP_PATH
exit
#REmove Temp Files
rm ${HYP_PATH} ${REF_PATH}

