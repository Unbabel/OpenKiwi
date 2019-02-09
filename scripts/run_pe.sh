#!/bin/bash

# additionally, you can specify all the --params here
# and they will overwrite the config file params
python3 kiwi train predest --config ../experiments/predictor_estimator.yaml \
                           --bad-weight 2.125 \
                           --rnn-layers 3

