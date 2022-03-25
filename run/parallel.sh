CONFIG_DIR=$1
REPEAT=$2
MAX_JOBS=${3:-2}
MAIN=${4:-main}

(
  trap 'kill 0' SIGINT
  CUR_JOBS=0
  for CONFIG in "$CONFIG_DIR"/*.yaml; do
    if [ "$CONFIG" != "$CONFIG_DIR/*.yaml" ]; then
      ((CUR_JOBS >= MAX_JOBS)) && wait -n
      echo "Job launched: $CONFIG"
      python $MAIN.py --cfg $CONFIG --repeat $REPEAT --mark_done &
      ((CUR_JOBS < MAX_JOBS)) && sleep 1
      ((++CUR_JOBS))
    fi
  done

  wait
)
