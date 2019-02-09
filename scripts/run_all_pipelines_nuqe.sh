#!/usr/bin/env bash

#SBATCH --gres=gpu:1

MODEL="nuqe"

DATASET="WMT18/word_level/en_de.nmt"
DATASET_NAME="WMT18"
FORMAT="wmt18"

OUTPUT_DIR_ROOT="data/predictions/${DATASET}"

JACKKNIFE_RUN_DIR_NAME="train"
TRAIN_RUN_DIR_NAME="dev"
PREDICT_RUN_DIR_NAME="test"

for SEED in 42 43 46 55 58
do

    OUTPUT_DIR_PREFIX="${OUTPUT_DIR_ROOT}/${MODEL}.seed${SEED}"

    for SIDE in "target" "gaps" "source"
    do
        JACKKNIFE_DIR="${OUTPUT_DIR_PREFIX}.${SIDE}/${JACKKNIFE_RUN_DIR_NAME}"
        TRAIN_DIR="${OUTPUT_DIR_PREFIX}.${SIDE}/${TRAIN_RUN_DIR_NAME}"
        PREDICT_DIR="${OUTPUT_DIR_PREFIX}.${SIDE}/${PREDICT_RUN_DIR_NAME}"

        python -m kiwi jackknife ${MODEL} --config experiments/nuqe/en_de/nuqe-${DATASET_NAME}-${SIDE}-jackknife.yaml \
                                         --seed ${SEED} \
                                         --gpu-id 0 \
                                         --output-dir ${JACKKNIFE_DIR}

        # Train
        python -m kiwi train ${MODEL} --config experiments/nuqe/en_de/nuqe-${DATASET_NAME}-${SIDE}.yaml \
                                     --mlflow-tracking-uri http://localhost:5000 \
                                     --seed ${SEED} \
                                     --gpu-id 0 \
                                     --output-dir ${TRAIN_DIR}

        # Predict
        python -m kiwi predict ${MODEL} --config experiments/nuqe/en_de/nuqe-${DATASET_NAME}-${SIDE}-predict.yaml \
                    --load-model ${TRAIN_DIR}/best_model.torch \
                    --load-vocab ${TRAIN_DIR}/vocab.torch \
                    --output-dir ${PREDICT_DIR}

    done


    # Merge gaps and target tags in a single file (as expected by evaluate.py)
    python scripts/merge_target_and_gaps_preds.py --target-pred ${OUTPUT_DIR_PREFIX}.target/${JACKKNIFE_RUN_DIR_NAME}/tags \
                                                  --gaps-pred ${OUTPUT_DIR_PREFIX}.gaps/${JACKKNIFE_RUN_DIR_NAME}/gap_tags \
                                                  --output ${OUTPUT_DIR_PREFIX}.targetgaps/${JACKKNIFE_RUN_DIR_NAME}/tags

    python scripts/merge_target_and_gaps_preds.py --target-pred ${OUTPUT_DIR_PREFIX}.target/${TRAIN_RUN_DIR_NAME}/epoch_*/tags \
                                                  --gaps-pred ${OUTPUT_DIR_PREFIX}.gaps/${TRAIN_RUN_DIR_NAME}/epoch_*/gap_tags \
                                                  --output ${OUTPUT_DIR_PREFIX}.targetgaps/${TRAIN_RUN_DIR_NAME}/tags

    python scripts/merge_target_and_gaps_preds.py --target-pred ${OUTPUT_DIR_PREFIX}.target/${PREDICT_RUN_DIR_NAME}/tags \
                                                  --gaps-pred ${OUTPUT_DIR_PREFIX}.gaps/${PREDICT_RUN_DIR_NAME}/gap_tags \
                                                  --output ${OUTPUT_DIR_PREFIX}.targetgaps/${PREDICT_RUN_DIR_NAME}/tags


    # Evaluate on dev set
    python scripts/evaluate.py --type probs  \
                               --format ${FORMAT} \
                               --gold-source data/${DATASET}/dev.src_tags \
                               --gold-target data/${DATASET}/dev.tags \
                               --pred-source ${OUTPUT_DIR_PREFIX}.source/${TRAIN_RUN_DIR_NAME}/epoch_*/source_tags \
                               --pred-target ${OUTPUT_DIR_PREFIX}.targetgaps/${TRAIN_RUN_DIR_NAME}/tags
    # Evaluate on test set
    python scripts/evaluate.py --type probs  \
                               --format ${FORMAT} \
                               --gold-source data/${DATASET}/test.src_tags \
                               --gold-target data/${DATASET}/test.tags \
                               --pred-source ${OUTPUT_DIR_PREFIX}.source/${PREDICT_RUN_DIR_NAME}/source_tags \
                               --pred-target ${OUTPUT_DIR_PREFIX}.targetgaps/${PREDICT_RUN_DIR_NAME}/tags

done

# Evaluate all predictions at once
python scripts/evaluate.py --type probs --format ${FORMAT} \
                           --gold-source data/${DATASET}/dev.src_tags \
                           --gold-target data/${DATASET}/dev.tags \
                           --pred-source ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.source/${TRAIN_RUN_DIR_NAME}/epoch_*/source_tags \
                           --pred-target ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.targetgaps/${TRAIN_RUN_DIR_NAME}/tags
python scripts/evaluate.py --type probs --format ${FORMAT} \
                           --gold-source data/${DATASET}/test.src_tags \
                           --gold-target data/${DATASET}/test.tags \
                           --pred-source ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.source/${PREDICT_RUN_DIR_NAME}/source_tags \
                           --pred-target ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.targetgaps/${PREDICT_RUN_DIR_NAME}/tags

# Prepare data for Linear
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/train.${MODEL}.stacked \
                                                 ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.target/${JACKKNIFE_RUN_DIR_NAME}/tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/train.${MODEL}.targetgaps.stacked \
                                                 ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.targetgaps/${JACKKNIFE_RUN_DIR_NAME}/tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/train.${MODEL}.source.stacked \
                                                 ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.source/${JACKKNIFE_RUN_DIR_NAME}/source_tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/train.${MODEL}.gaps.stacked \
                                                 ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.gaps/${JACKKNIFE_RUN_DIR_NAME}/gap_tags

python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/dev.${MODEL}.stacked ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.target/${TRAIN_RUN_DIR_NAME}/epoch_*/tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/dev.${MODEL}.targetgaps.stacked ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.targetgaps/${TRAIN_RUN_DIR_NAME}/tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/dev.${MODEL}.source.stacked ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.source/${TRAIN_RUN_DIR_NAME}/epoch_*/source_tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/dev.${MODEL}.gaps.stacked ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.gaps/${TRAIN_RUN_DIR_NAME}/epoch_*/gap_tags

python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/test.${MODEL}.stacked ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.target/${PREDICT_RUN_DIR_NAME}/tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/test.${MODEL}.targetgaps.stacked ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.targetgaps/${PREDICT_RUN_DIR_NAME}/tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/test.${MODEL}.source.stacked ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.source/${PREDICT_RUN_DIR_NAME}/source_tags
python scripts/stack_probabilities_for_linear.py -o ${OUTPUT_DIR_ROOT}/test.${MODEL}.gaps.stacked ${OUTPUT_DIR_ROOT}/${MODEL}.seed*.gaps/${PREDICT_RUN_DIR_NAME}/gap_tags
