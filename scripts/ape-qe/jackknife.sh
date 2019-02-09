#!/bin/bash

# this script preprocesses the training corpus, including tokenization,
# truecasing, and subword segmentation, for application to SRC->PE (distinct languages).


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
N_FOLDS=10

source ${SCRIPT_DIR}/env.sh

for i in $(seq 1 ${N_FOLDS}); do
    mkdir -p ${DATA_PATH}/fold_$i
done

python ${SCRIPT_DIR}/jackknife.py                   \
	     --data ${DATA_PATH}/$TRAIN             \
	     --out-dir ${DATA_PATH}                 \
	     --n-folds ${N_FOLDS}


for i in $(seq 1 ${N_FOLDS}); do
    python ${ONMT_PATH}/preprocess.py                                     \
	-train_src ${DATA_PATH}/fold_${i}/${PREFIX}.${TRAIN}.${SRC}       \
	-train_tgt ${DATA_PATH}/fold_${i}/${PREFIX}.${TRAIN}.${PE}        \
	-valid_src ${DATA_PATH}/${DEV}.${SRC}                             \
	-valid_tgt ${DATA_PATH}/${DEV}.${PE}                              \
	-save_data ${DATA_PATH}/fold_${i}/data
done

for i in $(seq 1 ${N_FOLDS}); do
    sbatch --job-name=fold_"$i" \
           --output=fold_"$i"   \
           --export=data=run_ape.sh ${DATA_PATH}/fold_${i}/data,model=${MODEL_PATH}/jackknife_${i}
done
