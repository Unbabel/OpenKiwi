#!/bin/bash

# Tokens to Classes

BAD_CLASS=BAD
OK_CLASS=OK

BAD_ID=1
OK_ID=0

INPUT=$1

sed -i "s/${OK_CLASS}/${OK_ID}/g" ${INPUT}
sed -i "s/${BAD_CLASS}/${BAD_ID}/g" ${INPUT}
