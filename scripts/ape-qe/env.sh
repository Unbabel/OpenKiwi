#paths
HOME="/home/sony"
export DATA_HOME="/mnt/data/home/sony"
export KIWI="/mnt/data/datasets/kiwi"
export MOSES_DECODER_PATH="${HOME}/mosesdecoder"
export SUBWORD_NMT_PATH="${HOME}/subword-nmt"
export ONMT_PATH="${HOME}/OpenNMT-py"
export TEREVAL="${HOME}/tercom-0.7.25/tercom.7.25.jar"
export MOSES_DECODER_PATH="${HOME}/mosesdecoder"

export DATA_PATH="${DATA_HOME}/ape/indomain/data/nmt"
export WMT_PATH="${KIWI}/WMT17/word_level/en_de"
export MODEL_PATH="${DATA_HOME}/ape/indomain/jackknife/nmt"

#preprocess settings
export TRAIN_TRUECASE=false
export APPLY_TRUECASE=true
export TRAIN_BPE=false
export APPLY_BPE=true
export OV=0

#languages
export L_SRC=en
export L_TRG=de

# suffix of files
export SRC=src
export PE=pe
export MT=mt

#prefixes
DEV=dev.bpe
TRAIN=train.bpe
export PREFIX_BIG=${DATA_PATH}/500K.bpe
export PREFIX_SMALL=${WMT_PATH}/train.bpe
export PREFIX_WMT=test
export PREFIX=500K.bpe

# truecase stem
TRUECASE_STEM=truecase-model


# number of merge operations. Network vocabulary should be slightly larger (to include characters),
# or smaller if the operations are learned on the joint vocabulary
bpe_operations=40000
