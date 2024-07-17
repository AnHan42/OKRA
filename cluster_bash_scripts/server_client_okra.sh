#!/bin/bash

#arguments $1 no of runs $2 OKRA/FLAKE $3 No Imput Party $4 base port 

# List of feature values
feature_values=("196608" "3145728")
#"192" "768" "3072" "12288"  "49152"

# Loop through feature values
for features in "${feature_values[@]}"
do
    # Loop for specified number of runs
    for i in $(seq 1 $1)
    do
        # Start Server.py
        base_port=$4
        port=$((base_port+i-1))

        python -u server_classification.py --num_clients $3 --base_port $port &

        sleep 4

        # Start clients
        for party_id in $(seq 1 $3)
        do
            echo "Starting client with arguments: --mask $2 --n_features $features --k 10 --party_id $party_id --n_samples 400 --base_port $port"
            python -u client_classification.py --mask $2 --n_features $features --k 10 --party_id $party_id --n_samples 400 --base_port $port --max_parties $parties &
	    sleep 0.1
        done

        wait
    done
done
