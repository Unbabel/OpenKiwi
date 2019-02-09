#!/bin/bash

# additionally, you can specify all the --params here
# and they will overwrite the config file params
python3 kiwi train quetch --config ../experiments/quetch-WMT17.yaml \
                          --window-size 7 \
                          --learning-rate 1.0
