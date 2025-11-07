#!/bin/bash

# Decrease number of experiments to run, as well as the number of timesteps, to test all experiment code

# Check if a filename is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    echo "Typical usage: $0 ."
    exit 1
fi

# Assign the first argument as the filename
FILENAME="$1"
base_dir="$1"

# Loop through every subdirectory in base_dir.
for dir in "$base_dir"/*/; do
  # Confirm that it's a directory.
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"
    # Loop through every file in the directory.
    for file in "$dir"*; do
      if [ -f "$file" ] && [[ "$file" == *.py || "$file" == *.slurm ]]; then
        echo "  Modifying file: $file"
        sed -i -e '/NUM_SAMPLES = /s/\(NUM_SAMPLES = \).*/\1 3/' "$file"
        sed -i -e '/^#SBATCH --nodes=/ s/\(^#SBATCH --nodes=\)[[:space:]]*.*/\1'"1"'/' "$file"
        sed -i "/config\['total_timesteps'\]/ s/\(config\['total_timesteps'\]\s*=\s*\).*/\1int(10E3)/" "$file"
      fi
    done
  fi
done

