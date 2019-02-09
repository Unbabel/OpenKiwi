BASE_DIR=/home/sony/multipred2
mkdir -p ${BASE_DIR}
COUNTER=0
COUNTER_BAD=0
for file in $(find /home/sony/OpenKiwi/runs/ -type f | grep model.torch); do
    #echo $file
    COUNTER_BAD=$((${COUNTER_BAD} + 1))
    logfile=$(dirname $(dirname $file))/output.log
    if [ -f $(dirname $file)/tags ] && [ -f $logfile ]; then
	echo $file
	COUNTER=$((${COUNTER} + 1))
	OUT_DIR=${BASE_DIR}/$COUNTER
	mkdir -p ${OUT_DIR}
	echo $logfile > ${OUT_DIR}/origin
	python kiwi predict --config /home/sony/OpenKiwi/experiments/predest/predict_18.yaml --model estimator --load-model $file --output-dir ${OUT_DIR} --gpu-id 0
	
    fi
done
echo "GOOD FILES: $COUNTER"
echo "TOTAL FILE: ${COUNTER_BAD}"
    

