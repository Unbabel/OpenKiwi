#!/bin/bash

# this script preprocesses the training corpus, including tokenization,
# truecasing, and subword segmentation, for application to SRC->PE (distinct languages).
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

source ${SCRIPT_DIR}/env.sh
source ${SCRIPT_DIR}/ape_utils.sh
if [[ $OV -gt 0 ]]; then
    echo "Oversampling..."
    python oversample.py $OV $PREFIX_BIG $PREFIX_SMALL $DATA_PATH/ov $SRC $PE
    PREFIX=ov
fi


echo "Tokenization and normalization..."
preprocess ${DATA_PATH}/$PREFIX.$SRC ${DATA_PATH}/$PREFIX.prep.$SRC $L_SRC
preprocess ${DATA_PATH}/$PREFIX.$PE ${DATA_PATH}/$PREFIX.prep.$PE $L_TRG


# train truecaser - only for src
if [[ ${TRAIN_TRUECASE} == "true" ]]; then
  echo "Training truecase model (only for source language)."
  ${MOSES_DECODER_PATH}/scripts/recaser/train-truecaser.perl -corpus ${DATA_PATH}/$PREFIX.prep.$SRC -model ${MODEL_PATH}/$TRUECASE_STEM.$L_SRC
  echo "Training truecase model (only for target language)."
  ${MOSES_DECODER_PATH}/scripts/recaser/train-truecaser.perl -corpus ${DATA_PATH}/$PREFIX.prep.$PE -model ${MODEL_PATH}/$TRUECASE_STEM.$L_TRG
fi
# apply truecaser (training corpus)
if [[ ${APPLY_TRUECASE} == "true" ]]; then
    echo "Applying truecasing."
    ${MOSES_DECODER_PATH}/scripts/recaser/truecase.perl -model ${MODEL_PATH}/$TRUECASE_STEM.$L_SRC < ${DATA_PATH}/$PREFIX.prep.$SRC > ${DATA_PATH}/$PREFIX.tc.$SRC
    ${MOSES_DECODER_PATH}/scripts/recaser/truecase.perl -model ${MODEL_PATH}/$TRUECASE_STEM.$L_TRG < ${DATA_PATH}/$PREFIX.prep.$PE > ${DATA_PATH}/$PREFIX.tc.$PE
    rm ${DATA_PATH}/$PREFIX.prep.{$PE,$SRC}    
fi


# train BPE - only for src
if [[ ${TRAIN_BPE} == "true" ]]; then
  echo "Training BPE model (only for source language)."
  ${SUBWORD_NMT_PATH}/learn_bpe.py -s $bpe_operations > ${MODEL_PATH}/$L_SRC.bpe < ${DATA_PATH}/$PREFIX.tc.$SRC
  echo "Training BPE model (only for target language)."
  ${SUBWORD_NMT_PATH}/learn_bpe.py -s $bpe_operations > ${MODEL_PATH}/$L_TRG.bpe < ${DATA_PATH}/$PREFIX.tc.$PE
fi

# apply BPE
if [[ ${APPLY_BPE} == "true" ]]; then
   echo "Applying BPE."
   ${SUBWORD_NMT_PATH}/apply_bpe.py -c ${MODEL_PATH}/$L_SRC.bpe < ${DATA_PATH}/$PREFIX.tc.$SRC > ${DATA_PATH}/$PREFIX.bpe.$SRC
   ${SUBWORD_NMT_PATH}/apply_bpe.py -c ${MODEL_PATH}/$L_TRG.bpe < ${DATA_PATH}/$PREFIX.tc.$PE > ${DATA_PATH}/$PREFIX.bpe.$PE
   rm ${DATA_PATH}/$PREFIX.tc.{$PE,$SRC}
fi
