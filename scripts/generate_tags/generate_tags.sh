function tag_to_id {
    # Tokens to Classes

    BAD_CLASS=BAD
    OK_CLASS=OK
    
    BAD_ID=1
    OK_ID=0
    
    INPUT=$1
    
    sed -i "s/${OK_CLASS}/${OK_ID}/g" ${INPUT}
    sed -i "s/${BAD_CLASS}/${BAD_ID}/g" ${INPUT}
}


CORPUS_GEN='/home/sony/word-level-qe-corpus-builder/corpus_generation'
SRC=$1
MT=$2
PE=$3
# Output directory
OUT=$4
# Fast Align Model To generate Source Tags
FA=$5
echo $FA
# 'true' or 'false'
WMT_18=$6

WD=$(pwd)
cd ${CORPUS_GEN}

TMP_DIR=__tmpdir__

bash get_tags.sh ${SRC} ${MT} ${PE} ${TMP_DIR}/ $FA ${WMT_18}
cd ${WD}
mkdir -p ${OUT}
# Remove first two lines and clip to 1.0 if greater than
cat ${CORPUS_GEN}/${TMP_DIR}/tercom/out_tercom_file.ter |
    awk '{if (NR!=1 && NR!=2) {if ($NF>1) print 1; else print $NF}}' > ${OUT}/hter
mv ${CORPUS_GEN}/${TMP_DIR}/tags ${OUT}/
tag_to_id ${OUT}/tags
if [[ ${WMT_18} == 'true' ]]; then
    mv ${CORPUS_GEN}/source_tags ${OUT}
    tag_to_id ${OUT}/source_tags
fi


