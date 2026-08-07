#!/bin/bash

# Stop script if critical error occurs
set -e

case "$(uname -s)" in
    Linux*)
        ACTIVATE_SCRIPT=".venv/bin/activate"
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        ACTIVATE_SCRIPT=".venv/Scripts/activate"
        ;;
    *)
        echo "Unsupported operating system: $(uname -s)"
        exit 1
        ;;
esac

source "$ACTIVATE_SCRIPT"

####################################################################################
# STREAK

# dim = 1
# N_VALUES=(5 10 20 30 50 70 100 150 200 250 300 500)
# DIM_VALUES=(1)

# dim = 2
# N_VALUES=(5 10 15 20 25 30 40 50 60 70 80 90 100)
# DIM_VALUES=(2)

# dim = 3
N_VALUES=(25 25 25 25 25 25 25 25 25 25)
# completed for 4, 6, 8, 10, 15
# N_VALUES=(4 6 8 10 15 25)
DIM_VALUES=(3)

STEPS=200_000

echo "Starting many simulations:"
echo "N_VALUES: ${N_VALUES[*]}"
echo "DIM_VALUES: ${DIM_VALUES[*]}"
# echo "STEPS: $STEPS"

start_time=$(date +%s)

for dim in "${DIM_VALUES[@]}"; do
    for N in "${N_VALUES[@]}"; do
        
        echo "* [$(date +'%Y-%m-%d %H:%M:%S')] Calling main.py with args: N=$N | dim=$dim | steps=$STEPS"
        
        python main.py -N "$N" -dim "$dim" -steps "$STEPS"
        
    done
done

current_hour=$(date +%H)

if [ "$current_hour" -lt 12 ]; then
    echo "* [$(date +'%Y-%m-%d %H:%M:%S')] It's before 12:00"
    echo "* [$(date +'%Y-%m-%d %H:%M:%S')] ==> Calling process.py (args are hardcoded inside the script) to filter and bake magnetization data for all simulations"
    
    python process.py
else
    echo "* [$(date +'%Y-%m-%d %H:%M:%S')] It's after 12:00"
    echo "* [$(date +'%Y-%m-%d %H:%M:%S')] ==> NOT executing process.py"
fi


end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
printf -v elapsed_hms '%02d:%02d:%02d' $((elapsed_seconds / 3600)) $(((elapsed_seconds % 3600) / 60)) $((elapsed_seconds % 60))

echo "[$(date +'%Y-%m-%d %H:%M:%S')] Completed in ${elapsed_hms}"
echo "--------------------------------------------------------"
