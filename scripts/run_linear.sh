#!/bin/bash

python3 kiwi train --config experiments/linear/en_de/linear.WMT17.yaml
python3 kiwi predict --config experiments/linear/en_de/linear.WMT17.predict.yaml
