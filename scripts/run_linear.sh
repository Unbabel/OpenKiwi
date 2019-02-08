#!/bin/bash

python3 kiwi train --config experiments/linear/en_de/linear.WMT18-SMT.yaml
python3 kiwi predict --config experiments/linear/en_de/linear.WMT18-SMT.predict.yaml
