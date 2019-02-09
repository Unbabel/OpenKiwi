FOLDER=$1

for partition in train dev test
do
    python3 extract_columns.py ${FOLDER}/additional/${partition}.mt.parsed 6 7 \
            > ${FOLDER}/additional/${partition}.mt.parses
    python3 extract_columns.py ${FOLDER}/features/${partition}.features 14 15 \
            > ${FOLDER}/additional/${partition}.mt.ngrams
done
