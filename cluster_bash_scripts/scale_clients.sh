#!/bin/bash

#arguments $1 no of runs $2 OKRA/FLAKE $3 base port

#fixed: 400 samples with 1000 features for every client

# List of feature values
no_inputparties=("5" "6" "7" "8" "9" "10")
# "3" "4" "50" "100" "200" "300" "400" "500"

# 10 random values for 10 runs
random_values=$(python -c "import random; random.seed(42); print(' '.join(map(str, [random.randint(1, 900) for _ in range(10)])))")

# Loop through feature values
for parties in "${no_inputparties[@]}"
do
    # Loop for specified number of runs
    for i in $(seq 1 $1)
		
    do        
	# get k
        k=$(echo $random_values | cut -d' ' -f$i)

        # Start Server.py
        base_port=$3
        port=$((base_port+i+100))

        python -u server_classification.py --num_clients $parties --base_port $port &

        sleep 4  

        # Start clients
        for party_id in $(seq 1 $parties)
        do
            python -u client_classification.py --mask $2 --n_features 1000 --k $k --party_id $party_id --n_samples 400 --base_port $port --max_parties $parties &
        sleep 0.1
        done

        wait
    done
done
