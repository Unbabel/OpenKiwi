#!/bin/bash

# additionally, you can specify all the --params here
# and they will overwrite the config file params
python3 kiwi train nuqe --config ../experiments/nuqe-WMT16.yaml \
                        --window-size 5 \
                        --learning-rate 0.1
