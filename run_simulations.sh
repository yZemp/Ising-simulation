#!/bin/bash

# Stop script if critical error occurs
set -e

case "$(uname -s)" in
    Linux*)
        ACTIVATE_SCRIPT="venv/bin/activate"
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        ACTIVATE_SCRIPT="venv/Scripts/activate"
        ;;
    *)
        echo "Unsupported operating system: $(uname -s)"
        exit 1
        ;;
esac

source "$ACTIVATE_SCRIPT"


N_VALUES=(10 20 30)
DIM_VALUES=(1)
STEPS=2_000

echo "Starting many simulations:"
echo "N_VALUES: ${N_VALUES[*]}"
echo "DIM_VALUES: ${DIM_VALUES[*]}"
echo "STEPS: $STEPS"

for dim in "${DIM_VALUES[@]}"; do
    for N in "${N_VALUES[@]}"; do
        
        echo "* [$(date +'%Y-%m-%d %H:%M:%S')] Simulating: N=$N | dim=$dim | steps=$STEPS"
        
        start_time=$(date +%s)
        python main.py -N "$N" -dim "$dim" -steps "$STEPS"
        end_time=$(date +%s)
        elapsed_seconds=$((end_time - start_time))
        printf -v elapsed_hms '%02d:%02d:%02d' $((elapsed_seconds / 3600)) $(((elapsed_seconds % 3600) / 60)) $((elapsed_seconds % 60))
        
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Completed in ${elapsed_hms}"
        echo "--------------------------------------------------------"
        
    done
done

echo "All simulations have been completed."
deactivate