#!/usr/bin/env bash

#SBATCH --gres=gpu:1

MODEL="nuqe"

LANGUAGE_PAIR="en_lv.nmt"
DATASET="WMT18/word_level/${LANGUAGE_PAIR}"
DATASET_NAME="wmt18"
FORMAT="wmt18"

CONFIGURATION_PATH=""

#OUTPUT_DIR_ROOT="data/predictions/${DATASET}"

JACKKNIFE_RUN_DIR_NAME="train"
TRAIN_RUN_DIR_NAME="dev"
PREDICT_RUN_DIR_NAME="test"

#data/predictions/WMT18/word_level/en_de.smt/nuqe/1/
#data/trained_models/release/wmt18.en_de.smt/nuqe/1/

OUTPUT_PREDICTIONS_ROOT_DIR="data/predictions/${DATASET}/${MODEL}"
OUTPUT_MODELS_ROOT_DIR="data/trained_models/release/${DATASET_NAME}.${LANGUAGE_PAIR}/${MODEL}"

COUNT=0
for SEED in 200 201 202 203 204
do
    COUNT=$((COUNT + 1))
    OUTPUT_PREDICTIONS_DIR="${OUTPUT_PREDICTIONS_ROOT_DIR}/${COUNT}"
    mkdir -p ${OUTPUT_PREDICTIONS_DIR}

#    OUTPUT_DIR_PREFIX="${OUTPUT_DIR_ROOT}/${MODEL}.seed${SEED}"

    echo "*****************************************************************"

    for SIDE in "target" "gaps" "source"
    do
        OUTPUT_MODEL_DIR="${OUTPUT_MODELS_ROOT_DIR}/${SIDE}/${COUNT}"

        JACKKNIFE_DIR="${OUTPUT_MODEL_DIR}/${JACKKNIFE_RUN_DIR_NAME}"
        TRAIN_DIR="${OUTPUT_MODEL_DIR}/${TRAIN_RUN_DIR_NAME}"
        PREDICT_DIR="${OUTPUT_MODEL_DIR}/${PREDICT_RUN_DIR_NAME}"

        echo "================================================================="
        if [[ ! -d "${JACKKNIFE_DIR:+$JACKKNIFE_DIR/}" ]]; then
            python -m kiwi jackknife --train-config experiments/nuqe/${DATASET_NAME}.${LANGUAGE_PAIR}/train-${SIDE}.yaml \
                                     --experiment-name "Official run for OpenKiwi" \
                                     --splits 10 \
                                     --seed ${SEED} \
                                     --gpu-id 0 \
                                     --output-dir ${JACKKNIFE_DIR}
            cp ${JACKKNIFE_DIR}/train*tags ${OUTPUT_PREDICTIONS_DIR}
        else
            echo "Skipping jackknifing; found ${JACKKNIFE_DIR}"
        fi

        # Train
        echo "================================================================="
        if [[ ! -d "${TRAIN_DIR:+$TRAIN_DIR/}" ]]; then
            python -m kiwi train --config experiments/nuqe/${DATASET_NAME}.${LANGUAGE_PAIR}/train-${SIDE}.yaml \
                                 --experiment-name "Official run for OpenKiwi" \
                                 --seed ${SEED} \
                                 --gpu-id 0 \
                                 --output-dir ${TRAIN_DIR}
            cp ${TRAIN_DIR}/epoch_*/tags ${OUTPUT_PREDICTIONS_DIR}/dev.tags
            cp ${TRAIN_DIR}/epoch_*/gap_tags ${OUTPUT_PREDICTIONS_DIR}/dev.gap_tags
            cp ${TRAIN_DIR}/epoch_*/src_tags ${OUTPUT_PREDICTIONS_DIR}/dev.src_tags
        else
            echo "Skipping training; found ${TRAIN_DIR}"
        fi

        # Predict
        echo "================================================================="
        python -m kiwi predict --config experiments/nuqe/${DATASET_NAME}.${LANGUAGE_PAIR}/predict-${SIDE}.yaml \
                               --experiment-name "Official run for OpenKiwi" \
                               --load-model ${TRAIN_DIR}/best_model.torch \
                               --output-dir ${PREDICT_DIR}
        cp ${PREDICT_DIR}/tags ${OUTPUT_PREDICTIONS_DIR}/test.tags
        cp ${PREDICT_DIR}/gap_tags ${OUTPUT_PREDICTIONS_DIR}/test.gap_tags
        cp ${PREDICT_DIR}/src_tags ${OUTPUT_PREDICTIONS_DIR}/test.src_tags

    done


    # Merge gaps and target tags in a single file (as expected by evaluate.py)
#    python scripts/merge_target_and_gaps_preds.py --target-pred ${OUTPUT_DIR_PREFIX}.target/${JACKKNIFE_RUN_DIR_NAME}/tags \
#                                                  --gaps-pred ${OUTPUT_DIR_PREFIX}.gaps/${JACKKNIFE_RUN_DIR_NAME}/gap_tags \
#                                                  --output ${OUTPUT_DIR_PREFIX}.targetgaps/${JACKKNIFE_RUN_DIR_NAME}/tags
#
#    python scripts/merge_target_and_gaps_preds.py --target-pred ${OUTPUT_DIR_PREFIX}.target/${TRAIN_RUN_DIR_NAME}/epoch_*/tags \
#                                                  --gaps-pred ${OUTPUT_DIR_PREFIX}.gaps/${TRAIN_RUN_DIR_NAME}/epoch_*/gap_tags \
#                                                  --output ${OUTPUT_DIR_PREFIX}.targetgaps/${TRAIN_RUN_DIR_NAME}/tags
#
#    python scripts/merge_target_and_gaps_preds.py --target-pred ${OUTPUT_DIR_PREFIX}.target/${PREDICT_RUN_DIR_NAME}/tags \
#                                                  --gaps-pred ${OUTPUT_DIR_PREFIX}.gaps/${PREDICT_RUN_DIR_NAME}/gap_tags \
#                                                  --output ${OUTPUT_DIR_PREFIX}.targetgaps/${PREDICT_RUN_DIR_NAME}/tags


    # Evaluate on dev set
    python scripts/evaluate.py --type probs  \
                               --format ${FORMAT} \
                               --gold-source data/${DATASET}/dev.src_tags \
                               --gold-target data/${DATASET}/dev.tags \
                               --pred-source ${OUTPUT_PREDICTIONS_DIR}/dev.src_tags \
                               --pred-gaps ${OUTPUT_PREDICTIONS_DIR}/dev.gap_tags \
                               --pred-target ${OUTPUT_PREDICTIONS_DIR}/dev.tags
    # Evaluate on test set
    python scripts/evaluate.py --type probs  \
                               --format ${FORMAT} \
                               --gold-source data/${DATASET}/test.src_tags \
                               --gold-target data/${DATASET}/test.tags \
                               --pred-source ${OUTPUT_PREDICTIONS_DIR}/test.src_tags \
                               --pred-gaps ${OUTPUT_PREDICTIONS_DIR}/test.gap_tags \
                               --pred-target ${OUTPUT_PREDICTIONS_DIR}/test.tags

done

# Evaluate all predictions at once
python scripts/evaluate.py --type probs --format ${FORMAT} \
                           --gold-source data/${DATASET}/dev.src_tags \
                           --gold-target data/${DATASET}/dev.tags \
                           --pred-source ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/dev.src_tags \
                           --pred-gaps ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/dev.gap_tags \
                           --pred-target ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/dev.tags
python scripts/evaluate.py --type probs --format ${FORMAT} \
                           --gold-source data/${DATASET}/test.src_tags \
                           --gold-target data/${DATASET}/test.tags \
                           --pred-source ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/test.src_tags \
                           --pred-gaps ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/test.gap_tags \
                           --pred-target ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/test.tags

# Prepare data for Linear
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_PREDICTIONS_ROOT_DIR}/train.${MODEL}.stacked ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/train.tags

python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_PREDICTIONS_ROOT_DIR}/dev.${MODEL}.stacked ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/dev.tags

python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_PREDICTIONS_ROOT_DIR}/test.${MODEL}.stacked ${OUTPUT_PREDICTIONS_ROOT_DIR}/[1-${COUNT}]/test.tags
