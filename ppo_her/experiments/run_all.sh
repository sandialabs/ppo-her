#!/bin/bash

# Get the current directory
BASE_DIR=$(pwd)

# Loop through each subdirectory to run optimize.slurm and process.slurm
for dir in "$BASE_DIR"/*/; do
    if [ -d "$dir" ] && [ ! -f "$dir/success.txt" ]; then
        echo "Entering directory: $dir"
        cd "$dir" || exit  # Change to the directory, exit if it fails

        # Run the optimize.slurm script
        if [ -f "optimize.slurm" ]; then
            echo "Running optimize.slurm in $dir"
            OPTIMIZE_JOB_ID=$(sbatch optimize.slurm | awk '{print $4}')  # Capture the job ID
            echo "Submitted optimize.slurm with Job ID: $OPTIMIZE_JOB_ID"

            # Run the process.slurm script with dependency on optimize.slurm
            if [ -f "process.slurm" ]; then
                echo "Running process.slurm in $dir with dependency on Job ID: $OPTIMIZE_JOB_ID"
                sbatch --dependency=afterok:$OPTIMIZE_JOB_ID process.slurm
            else
                echo "process.slurm not found in $dir"
            fi
        else
            echo "optimize.slurm not found in $dir"
        fi

        # Optionally, return to the base directory
        cd "$BASE_DIR" || exit  # Change back to the base directory, exit if it fails
    fi
done