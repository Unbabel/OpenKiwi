predictor1=$1
predictor2=$2
path=/mnt/data/datasets/kiwi/predictions/WMT17/word_level/en_de

for partition in train dev test
do
    echo $partition
    paste -d "|" $path/$partition.$predictor1.stacked \
        $path/$partition.$predictor2.stacked | sed "s/^|$//g" \
        > $path/$partition.${predictor1}_${predictor2}.stacked
done
