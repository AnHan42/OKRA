#!/bin/bash

#arguments $1 no of runs $2 base port

# List of feature values
#no_inputparties=("5" "6" "7" "8" "9" "10")
no_inputparties=("3")

#no_features=("128" "512" "2048" "8192" "32768" "131072" "524288")
no_features=("512")

# privacy parameter k: 10 random values for 10 runs
#random_values=$(python -c "import random; random.seed(42); print(' '.join(map(str, [random.randint(1, 900) for _ in range(10)])))")

# Loop through feature values
for parties in "${no_inputparties[@]}"
do
    # Loop for specified number of runs
    for i in $(seq 1 $1)
		
    do        
		# get k
		#k=$(echo $random_values | cut -d' ' -f$i)

        # Start Server.py
        base_port=$2
        port=$((base_port+i+100))

        python -u server_classification.py --num_clients $parties --base_port $port --n_samples 400 &

        sleep 4 

        # Start clients
        for party_id in $(seq 1 $parties)
        do
            python -u client_classification.py --n_features 30 --k 1 --party_id $party_id --n_samples 400 --base_port $port --max_parties $parties
            sleep 0.1 
        done

        wait
    done
done
