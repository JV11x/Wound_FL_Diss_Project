#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
python server.py > server.log 2>&1 &
sleep 10  # Sleep for 10s to give the server enough time to start

# count the number of clients
num_clients=6

for i in `seq 1 $num_clients`; do
    echo "Starting client $i"
    CLIENT_ID=$i TOTAL_CLIENTS=$num_clients python client.py > "client$i.log" 2>&1 &
done

# Enable CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
