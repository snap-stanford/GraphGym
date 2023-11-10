ESET=$1
PATIENT=$2
CYTO=$3
GRID=$4

python generation.py ${ESET} ${PATIENT} ${CYTO} ${GRID}
wait


if [ $GRID = "True" ]; then
    bash run_custom_batch_${ESET}_${CYTO}.sh
elif [ $GRID = "False" ]; then
    bash run_custom_${ESET}_${CYTO}.sh
fi