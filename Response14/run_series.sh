#!/bin/bash

# Initial input size
input_size=3

# Run the loop 9 times
for i in {1..9}
do
   echo "Running with input size: $input_size"
   python ShallowCNN.py $input_size
   python plotShallowCNN.py $input_size 
   # Double the input size for the next iteration
   input_size=$((input_size * 2))
done
