#!/bin/bash

# this script preprocesses the training corpus, including tokenization,
# truecasing, and subword segmentation, for application to SRC->PE (distinct languages).


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
N_FOLDS=10
source ${SCRIPT_DIR}/env.sh
source ${SCRIPT_DIR}/ape_utils.sh


source ${SCRIPT_DIR}/env.sh

for i in $(seq 1 ${N_FOLDS}); do
    MODEL=${MODEL_PATH}/fold_${i}.pt
    mv ${MODEL_PATH}/jackknife_${i}_step_125000.pt ${MODEL}
    rm -rf ${MODEL_PATH}/jackknife_${i}_step*
    FOLD_PATH=${DATA_PATH}/fold_${i}
    SRC_PATH=${FOLD_PATH}/pred_fold.src
    OUT_PATH=${DATA_PATH}/pred_${i}
    srun --gres=gpu:1 --partition=medium   \
	 python ${ONMT_PATH}/translate.py  \
	 -gpu 0                            \
	 -model ${MODEL}                   \
	 -src ${SRC_PATH}                  \
	 -output ${OUT_PATH}               \
	 -length_penalty wu                \
	 -alpha 0.7                        \
	 -replace_unk                      \
	 -beam_size 12                     \
	 -verbose
done

cat $(eval echo ${DATA_PATH}/pred_{1..${N_FOLDS}})  > ${DATA_PATH}/pred_full

remove_bpe ${DATA_PATH}/pred_full

#tercom ${JACKKNIFE}/pred_full
