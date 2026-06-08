#!/bin/bash

# Configuration — override by exporting before running, e.g.:
#   export competitions="titanic spaceship_titanic"
#   export start_run=1 end_run=3 model="llama-3.1-8b-instant"
#   bash run_multi_agent.sh

competitions=(${competitions:-titanic})
start_run=${start_run:-1}
end_run=${end_run:-5}
dest_dir_param=${dest_dir_param:-all_tools}
model=${model:-llama-3.1-8b-instant}

for competition in "${competitions[@]}"; do
    for run in $(seq $start_run $end_run); do
        echo "Running $competition, iteration $run"
        python multi_agents/sop.py \
            --competition "$competition" \
            --start_phase 1 \
            --end_phase 6 \
            --run "$run" \
            --dest_dir "$dest_dir_param" \
            --model "$model"
    done
done
