#!/bin/bash

# this script preprocesses the WMT corpus (Truecase and optional BPE)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
source "${SCRIPT_DIR}/env.sh"

# apply truecaser
echo "Applying truecasing."

$MOSES_DECODER_PATH/scripts/recaser/truecase.perl -model ${MODEL_PATH}/${TRUECASE_STEM}.${L_SRC} < ${WMT_PATH}/${PREFIX_WMT}.$SRC > ${DATA_PATH}/${PREFIX_WMT}.tc.$SRC
# $MOSES_DECODER_PATH/scripts/recaser/truecase.perl -model ${MODEL_PATH}/${TRUECASE_STEM}.${L_TRG} < ${WMT_PATH}/${PREFIX_WMT}.$PE > ${DATA_PATH}/${PREFIX_WMT}.tc.$PE
# $MOSES_DECODER_PATH/scripts/recaser/truecase.perl -model ${MODEL_PATH}/${TRUECASE_STEM}.${L_TRG} < ${WMT_PATH}/${PREFIX_WMT}.$MT > ${DATA_PATH}/${PREFIX_WMT}.tc.$MT


# apply BPE
if [[ ${APPLY_BPE} == "true" ]]; then
    echo "Applying BPE."
    $SUBWORD_NMT_PATH/apply_bpe.py -c ${MODEL_PATH}/$L_SRC.bpe < ${DATA_PATH}/${PREFIX_WMT}.tc.$SRC > ${DATA_PATH}/${PREFIX_WMT}.bpe.$SRC
#    $SUBWORD_NMT_PATH/apply_bpe.py -c ${MODEL_PATH}/$L_TRG.bpe < ${DATA_PATH}/${PREFIX_WMT}.tc.$PE > ${DATA_PATH}/${PREFIX_WMT}.bpe.$PE
#    $SUBWORD_NMT_PATH/apply_bpe.py -c ${MODEL_PATH}/$L_TRG.bpe < ${DATA_PATH}/${PREFIX_WMT}.tc.$MT > ${DATA_PATH}/${PREFIX_WMT}.bpe.$MT
    rm ${DATA_PATH}/${PREFIX_WMT}.tc.{$SRC,$PE,$MT}
fi

