#!/bin/env bash

export LD_LIBRARY_PATH=/app/bin/lib/:$LD_LIBRARY_PATH


cd /app/bin && /app/bin/start.sh \
    -t $TEAM_NAME \
    --host $HOST \
    --port $PORT &

cd /app/code && python3 /app/code/grpc-server.py

