#!/bin/bash

# Array of program names
programs=("program11.out" "program12.out" "program13.out" "program14.out" "program10.out")

# Loop through each program
for prog in "${programs[@]}"; do
    # Define the output file name based on the program name
    output_name="${prog%.out}.qdstrm"

    # Profile the program
    echo "Profiling $prog..."
    nsys profile --stats=true --output=$output_name ./$prog

    # Run the program
    echo "Running $prog..."
    ./$prog
done
