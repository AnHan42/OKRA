#!/bin/bash

#arguments $1 no of runs

# Loop i times
for i in $(seq 1 $1)
do

# Start Server.py
base_port=8000
base_features=200
port=$((base_port+i-1))
features=$((base_features*i))

python server_classification.py --num_clients 3 --base_port $port --FL False &

sleep 2

# Start Alice
python client_classification.py --mask "FLAKE" --n_features $features --k 10 --party_id 1 --n_samples 200 --base_port $port &

sleep 2

# Start Bob
python client_classification.py --mask "FLAKE" --n_features $features --k 10 --party_id 2 --n_samples 200 --base_port $port &

#sleep 2

# Start Charlie
python client_classification.py --mask "FLAKE" --n_features $features --k 10 --party_id 3 --n_samples 200 --base_port $port &


wait

done
