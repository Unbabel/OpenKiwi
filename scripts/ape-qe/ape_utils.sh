# Remove BPE segmentation
function remove_bpe {
    sed -i 's/\@\@ //g' $1
}

# Append Line numbers in parentheses to output - required by tercom
function append_line_numbers {
    PATH_SRC=$1
    PATH_OUT=$2
    awk '{print $0 " (" NR  ")"}' ${PATH_SRC} > ${PATH_OUT}
}

# Score using Tercom
function tercom {
    GOLD_PATH=$1
    PRED_PATH=$2
    append_line_numbers ${PRED_PATH} ${PRED_PATH}.tmp
    append_line_numbers ${GOLD_PATH} ${GOLD_PATH}.tmp
    OUT_PATH="$(dirname ${PRED_PATH})/$(basename ${GOLD_PATH})_$(basename ${PRED_PATH})"
    java -jar ${TEREVAL} -s -r ${GOLD_PATH}.tmp -h ${PRED_PATH}.tmp -n ${OUT_PATH}
    rm ${PRED_PATH}.tmp ${GOLD_PATH}.tmp
}

function preprocess {
    FILE=$1
    OUT_FILE=$2
    LANG=$3
    echo "Processing $FILE ..."
    cat $FILE | \
	$MOSES_DECODER_PATH/scripts/tokenizer/detokenizer.perl -l $LANG | \
	$MOSES_DECODER_PATH/scripts/tokenizer/normalize-punctuation.perl -l $LANG | \
	$MOSES_DECODER_PATH/scripts/tokenizer/tokenizer.perl -no-escape -l $LANG -threads 8 > ${OUT_FILE}
}

function postprocess {
    FILE=$1
    OUT_FILE=$FILE.tmp
    remove_bpe $FILE
    cat $FILE | \
	$MOSES_DECODER_PATH/scripts/recaser/detruecase.perl > ${OUT_FILE}
    mv ${OUT_FILE} $FILE
}

function translate {
    MODEL_PATH=$1
    SRC_PATH=$2
    OUT_PATH=$3
    GPU_ID=$4
    python ${ONMT_PATH}/translate.py  \
	   -model ${MODEL_PATH}       \
	   -src ${SRC_PATH}           \
	   -output ${OUT_PATH}        \
	   -gpu ${GPU_ID}             \
	   -length_penalty wu         \
	   -alpha 0.7                 \
	   -replace_unk               \
	   -beam_size 12              \
	   -verbose
}

function translate_ape {
    MODEL_PATH=$1
    SRC_PATH=$2
    MT_PATH=$3
    OUT_PATH=$4
    GPU_ID=$5
    python ${ONMT_PATH}/translate.py  \
	   -model ${MODEL_PATH}       \
	   -src ${SRC_PATH}           \
	   -mt ${MT_PATH}             \
	   -output ${OUT_PATH}        \
	   -gpu ${GPU_ID}             \
	   -length_penalty wu         \
	   -alpha 0.7                 \
	   -replace_unk               \
	   -beam_size 12              \
	   -verbose
}

