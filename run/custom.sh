ESET=$1
PATIENT=$2
CYTO=$3
GRID=$4

python generation.py ${ESET} ${PATIENT} ${CYTO} ${GRID}
wait

bash run_custom_${ESET}_${CYTO}.sh