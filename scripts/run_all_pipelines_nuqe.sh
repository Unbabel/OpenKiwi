#!/usr/bin/env bash

#SBATCH --gres=gpu:1

GPU=0
MODEL="nuqe"

<<<<<<< HEAD
LANGUAGE_PAIR="en_lv.smt"
=======
LANGUAGE_PAIR="de_en.smt"
>>>>>>> 1548995... Use correct embeddings for lv and cs and fix loading NuQE source model
DATASET="WMT18/word_level/${LANGUAGE_PAIR}"
DATASET_NAME="wmt18"
FORMAT="wmt18"

# e.g.: data/predictions/WMT18/word_level/en_de.smt/nuqe/1/
OUTPUT_PREDICTIONS_ROOT_DIR="data/predictions/${DATASET}/${MODEL}"

# e.g.: data/trained_models/release/wmt18.en_de.smt/nuqe/1/
OUTPUT_MODELS_ROOT_DIR="data/trained_models/release/${DATASET_NAME}.${LANGUAGE_PAIR}/${MODEL}"

JACKKNIFE_RUN_DIR_NAME="train"
TRAIN_RUN_DIR_NAME="dev"
PREDICT_RUN_DIR_NAME="test"

RUN_JACKKNIFE=false

COUNT=0
for SEED in 200 #201 202 203 204
do
    COUNT=$((COUNT + 1))
    OUTPUT_PREDICTIONS_DIR="${OUTPUT_PREDICTIONS_ROOT_DIR}/${COUNT}"
    mkdir -p ${OUTPUT_PREDICTIONS_DIR}

    echo "*****************************************************************"

    for SIDE in "target" "source" "gaps"
    do
        OUTPUT_MODEL_DIR="${OUTPUT_MODELS_ROOT_DIR}/${SIDE}/${COUNT}"

        JACKKNIFE_DIR="${OUTPUT_MODEL_DIR}/${JACKKNIFE_RUN_DIR_NAME}"
        TRAIN_DIR="${OUTPUT_MODEL_DIR}/${TRAIN_RUN_DIR_NAME}"
        PREDICT_DIR="${OUTPUT_MODEL_DIR}/${PREDICT_RUN_DIR_NAME}"

        echo "================================================================="
<<<<<<< HEAD
        if ${RUN_JACKKNIFE}
        then
            if [[ ! -d "${JACKKNIFE_DIR:+$JACKKNIFE_DIR/}" ]]
            then
                python -m kiwi jackknife --train-config experiments/nuqe/${DATASET_NAME}.${LANGUAGE_PAIR}/train-${SIDE}.yaml \
                                         --experiment-name "Official run for OpenKiwi" \
                                         --splits 10 \
                                         --seed ${SEED} \
                                         --gpu-id ${GPU} \
                                         --output-dir ${JACKKNIFE_DIR}
                cp ${JACKKNIFE_DIR}/train*tags ${OUTPUT_PREDICTIONS_DIR}
            else
                echo "Skipping jackknifing; found ${JACKKNIFE_DIR}"
            fi
=======
        if ${RUN_JACKKNIFE} && [[ ! -d "${JACKKNIFE_DIR:+$JACKKNIFE_DIR/}" ]]; then
            python -m kiwi jackknife --train-config experiments/nuqe/${DATASET_NAME}.${LANGUAGE_PAIR}/train-${SIDE}.yaml \
                                     --experiment-name "Official run for OpenKiwi" \
                                     --splits 10 \
                                     --seed ${SEED} \
                                     --gpu-id ${GPU} \
                                     --output-dir ${JACKKNIFE_DIR}
            cp ${JACKKNIFE_DIR}/train*tags ${OUTPUT_PREDICTIONS_DIR}
>>>>>>> 1548995... Use correct embeddings for lv and cs and fix loading NuQE source model
        else
            echo "RUN_JACKKNIFE is false"
        fi

        if [[ $SIDE = "target" ]]
        then
            TAGS_FILE="tags"
        elif [[ ${SIDE} = "gaps" ]]
        then
            TAGS_FILE="gap_tags"
        else
            TAGS_FILE="source_tags"
        fi

        # Train
        echo "================================================================="
        if [[ ! -d "${TRAIN_DIR:+$TRAIN_DIR/}" ]]; then
            python -m kiwi train --config experiments/nuqe/${DATASET_NAME}.${LANGUAGE_PAIR}/train-${SIDE}.yaml \
                                 --experiment-name "Official run for OpenKiwi" \
                                 --seed ${SEED} \
                                 --gpu-id ${GPU} \
                                 --checkpoint-keep-only-best 1 \
                                 --output-dir ${TRAIN_DIR}
        else
            echo "Skipping training; found ${TRAIN_DIR}"
        fi
#            cp ${TRAIN_DIR}/epoch_*/${TAGS_FILE} ${OUTPUT_PREDICTIONS_DIR}/dev.${TAGS_FILE}
<<<<<<< HEAD
        cp ${TRAIN_DIR}/epoch_*/tags ${OUTPUT_PREDICTIONS_DIR}/dev.tags 1> /dev/null 2>&1
        cp ${TRAIN_DIR}/epoch_*/gap_tags ${OUTPUT_PREDICTIONS_DIR}/dev.gap_tags 1> /dev/null 2>&1
        cp ${TRAIN_DIR}/epoch_*/source_tags ${OUTPUT_PREDICTIONS_DIR}/dev.source_tags 1> /dev/null 2>&1
=======
        if [[ -f "${TRAIN_DIR}/epoch_*/tags" ]]; then cp ${TRAIN_DIR}/epoch_*/tags ${OUTPUT_PREDICTIONS_DIR}/dev.tags; fi
        if [[ -f "${TRAIN_DIR}/epoch_*/gap_tags" ]]; then cp ${TRAIN_DIR}/epoch_*/gap_tags ${OUTPUT_PREDICTIONS_DIR}/dev.gap_tags; fi
        if [[ -f "${TRAIN_DIR}/epoch_*/source_tags" ]]; then cp ${TRAIN_DIR}/epoch_*/source_tags ${OUTPUT_PREDICTIONS_DIR}/dev.source_tags; fi
>>>>>>> 1548995... Use correct embeddings for lv and cs and fix loading NuQE source model

        # Predict
        echo "================================================================="
        python -m kiwi predict --config experiments/nuqe/${DATASET_NAME}.${LANGUAGE_PAIR}/predict-${SIDE}.yaml \
                               --experiment-name "Official run for OpenKiwi" \
                               --load-model ${TRAIN_DIR}/best_model.torch \
                               --output-dir ${PREDICT_DIR}
<<<<<<< HEAD
        cp ${PREDICT_DIR}/tags ${OUTPUT_PREDICTIONS_DIR}/test.tags 1> /dev/null 2>&1
        cp ${PREDICT_DIR}/gap_tags ${OUTPUT_PREDICTIONS_DIR}/test.gap_tags 1> /dev/null 2>&1
        cp ${PREDICT_DIR}/source_tags ${OUTPUT_PREDICTIONS_DIR}/test.source_tags 1> /dev/null 2>&1
=======
        if [[ -f "${PREDICT_DIR}/epoch_*/tags" ]]; then cp ${PREDICT_DIR}/epoch_*/tags ${OUTPUT_PREDICTIONS_DIR}/test.tags; fi
        if [[ -f "${PREDICT_DIR}/epoch_*/gap_tags" ]]; then cp ${PREDICT_DIR}/epoch_*/gap_tags ${OUTPUT_PREDICTIONS_DIR}/test.gap_tags; fi
        if [[ -f "${PREDICT_DIR}/epoch_*/source_tags" ]]; then cp ${PREDICT_DIR}/epoch_*/source_tags ${OUTPUT_PREDICTIONS_DIR}/test.source_tags; fi
>>>>>>> 1548995... Use correct embeddings for lv and cs and fix loading NuQE source model

    done


    # Evaluate on dev set
    python scripts/evaluate.py --type probs  \
                               --format ${FORMAT} \
                               --gold-source data/${DATASET}/dev.src_tags \
                               --gold-target data/${DATASET}/dev.tags \
                               --pred-format wmt17 \
                               --pred-source ${OUTPUT_PREDICTIONS_DIR}/dev.source_tags \
                               --pred-gaps ${OUTPUT_PREDICTIONS_DIR}/dev.gap_tags \
                               --pred-target ${OUTPUT_PREDICTIONS_DIR}/dev.tags
    # Evaluate on test set
    python scripts/evaluate.py --type probs  \
                               --format ${FORMAT} \
                               --gold-source data/${DATASET}/test.src_tags \
                               --gold-target data/${DATASET}/test.tags \
                               --pred-format wmt17 \
                               --pred-source ${OUTPUT_PREDICTIONS_DIR}/test.source_tags \
                               --pred-gaps ${OUTPUT_PREDICTIONS_DIR}/test.gap_tags \
                               --pred-target ${OUTPUT_PREDICTIONS_DIR}/test.tags

done

if [[ ${COUNT} -gt 1 ]]
then
    # Evaluate all predictions at once
    python scripts/evaluate.py --type probs --format ${FORMAT} \
                               --gold-source data/${DATASET}/dev.src_tags \
                               --gold-target data/${DATASET}/dev.tags \
                               --pred-format wmt17 \
                               --pred-source ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/dev.source_tags \
                               --pred-gaps ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/dev.gap_tags \
                               --pred-target ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/dev.tags
    python scripts/evaluate.py --type probs --format ${FORMAT} \
                               --gold-source data/${DATASET}/test.src_tags \
                               --gold-target data/${DATASET}/test.tags \
                               --pred-format wmt17 \
                               --pred-source ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/test.source_tags \
                               --pred-gaps ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/test.gap_tags \
                               --pred-target ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/test.tags
fi

ALL_RUNS=${COUNT}
if [[ ${COUNT} -gt 1 ]]
then
    ALL_RUNS="[1-${COUNT}]"
fi

# Prepare data for Linear
if ${RUN_JACKKNIFE}
then
    python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_PREDICTIONS_ROOT_DIR}/train.${MODEL}.stacked ${OUTPUT_PREDICTIONS_ROOT_DIR}/${ALL_RUNS}/train.tags
fi
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_PREDICTIONS_ROOT_DIR}/dev.${MODEL}.stacked ${OUTPUT_PREDICTIONS_ROOT_DIR}/${ALL_RUNS}/dev.tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_PREDICTIONS_ROOT_DIR}/test.${MODEL}.stacked ${OUTPUT_PREDICTIONS_ROOT_DIR}/${ALL_RUNS}/test.tags

