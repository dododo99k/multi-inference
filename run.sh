#!/bin/bash
for i in {1..3}; do
    echo "Running command $i..."
    python inference_baseline.py &  
done
