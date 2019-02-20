BASE_DIR=/home/sony/multipred2
1mkdir -p ${BASE_DIR}
COUNTER=0
COUNTER_BAD=0
WMT18=/mnt/data/datasets/kiwi/WMT18
ARGS="--gold-target ${WMT18}/word_level/en_de.nmt/dev.tags --gold-sents ${WMT18}/sentence_level/en_de.nmt/dev.hter"
SENTS="--pred-sents"
TAGS="--pred-target"
GAPS="--pred-gaps"
high=0.574
low=0.3

for folder in $(find ${BASE_DIR} -type d); do
    tag="$folder/tags"
    sent="$folder/sentence_scores"
    gap="$folder/gap_tags"
    log=$(cat $folder/origin)


    
    if [ -f $tag ]; then
	echo TAG
	fmul=$(cat $log | grep "EVAL_tags_F1_MULT" | tail -n1 |
		      awk '{ for (x=1;x<=NF;x++) if ($x~"EVAL_tags_F1_MULT") print $(x+1) }' |
		      sed s/,//g)
	echo $fmul
	if [ $(echo  "$fmul > 0.5" | bc) -eq 1 ]; then
	    freq=$(echo "scale=5; ($fmul-0.43)/0.02" | bc | xargs -I{} echo "{} / 1" | bc)
	    echo $freq
	    for i in $(seq 1 $freq); do
		TAGS="$TAGS $tag"
	    done
	fi
    fi
    if [ -f $sent ]; then
	echo SENT
	pearson=$(cat $log | grep "EVAL_PEARSON" | tail -n1 |
			 awk '{ for (x=1;x<=NF;x++) if ($x~"EVAL_PEARSON") print $(x+1) }' |
			 sed s/,//g)
	echo $pearson
	if [ $(echo "$pearson > 0.5" | bc ) -eq 1 ]; then
	    freq=$(echo "scale=5; ($pearson - 0.48)/0.02" | bc | xargs -I{} echo "{} / 1" | bc)
	    echo $freq
	    for i in $(seq 1 $freq); do
		SENTS="$SENTS $sent"
	    done
	fi
    fi
    
    if [ -f $gap ]; then
	echo GAPS
	fmul=$(cat $log | grep "EVAL_gap_tags_F1_MULT" | tail -n1 |
		      awk '{ for (x=1;x<=NF;x++) if ($x~"EVAL_gap_tags_F1_MULT") print $(x+1) }' |
		      sed s/,//g)
	echo $fmul
	if [[ $(echo "$fmul > 0.35" | bc) -eq 1 ]]; then
	    freq=$(echo "scale=5; ($fmul-0.35)/0.02" | bc | xargs -I{} echo "{} / 1" | bc)
	    echo $freq
	    for i in $(seq 1 $freq); do
		GAPS="$GAPS $gap"
	    done
	fi
	GAPS="$GAPS $gap"
    fi
done

ARGS="$ARGS $SENTS $GAPS $TAGS --format wmt18 --pred-format wmt18  --type probs"
kiwi evaluate $ARGS
