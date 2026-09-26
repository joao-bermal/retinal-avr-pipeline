#!/bin/bash
# GPU temperature watchdog for training on AMD/ROCm (see docs/GPU_ROCM_RX6800XT.md).
#
# Runs a command in the background, polls the GPU junction temperature with
# rocm-smi, and kills the command if it reaches the limit, well before the
# card's own critical shutdown threshold (110 C / 115 C on the RX 6800XT).
#
# Usage:
#   scripts/temp_guard.sh "<command>" <logfile>
#   LIMIT_C=80 CHECK_INTERVAL=2 scripts/temp_guard.sh ".venv/bin/python main.py --train_od" logs/train_od.log
#
# Writes the command output to <logfile> and one temperature reading per
# poll to <logfile>.temp. Exits 1 if the watchdog aborted the run, otherwise
# with the command's own exit code.
set -u

if [ $# -lt 2 ]; then
    echo "usage: $0 \"<command>\" <logfile>" >&2
    exit 2
fi

CMD="$1"
LOGFILE="$2"
LIMIT_C="${LIMIT_C:-85}"
CHECK_INTERVAL="${CHECK_INTERVAL:-1}"

mkdir -p "$(dirname "$LOGFILE")"
bash -c "$CMD" > "$LOGFILE" 2>&1 &
PID=$!
echo "watchdog: pid=$PID limit=${LIMIT_C}C interval=${CHECK_INTERVAL}s log=$LOGFILE"

while kill -0 "$PID" 2>/dev/null; do
    TEMP=$(rocm-smi --showtemp --json 2>/dev/null \
        | grep -o '"Temperature (Sensor junction) (C)": "[0-9.]*"' \
        | grep -o '[0-9.]*' | head -1)
    if [ -n "$TEMP" ]; then
        TEMP_INT=${TEMP%.*}
        echo "$(date '+%H:%M:%S') junction=${TEMP}C" >> "${LOGFILE}.temp"
        if [ "$TEMP_INT" -ge "$LIMIT_C" ]; then
            echo "ABORTING: junction temperature ${TEMP}C >= limit ${LIMIT_C}C" | tee -a "$LOGFILE" "${LOGFILE}.temp"
            kill -TERM "$PID" 2>/dev/null
            sleep 5
            kill -KILL "$PID" 2>/dev/null
            exit 1
        fi
    fi
    sleep "$CHECK_INTERVAL"
done

wait "$PID"
STATUS=$?
echo "watchdog: command exited with status $STATUS"
exit $STATUS
