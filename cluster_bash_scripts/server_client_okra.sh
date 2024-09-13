#!/bin/bash

#arguments $1 no of runs $32 No Imput Party $3 base port 

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
        base_port=$3
        port=$((base_port+i-1))

        python -u server_classification.py --num_clients $2 --base_port $port &

        sleep 4

        # Start clients
        for party_id in $(seq 1 $2)
        do
            echo "Starting client with arguments: --n_features $features --k 10 --party_id $party_id --n_samples 400 --base_port $port"
            python -u client_classification.py --n_features $features --k 10 --party_id $party_id --n_samples 400 --base_port $port --max_parties $parties &
	    sleep 0.1
        done

        wait
    done
done
