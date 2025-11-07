#!/bin/bash

#SBATCH --time=1:00:00
#SBATCH --job-name=herType_proc
#SBATCH --output=%j.log

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=1
#SBATCH --exclusive

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1

# Load modules or your own conda environment here
SCRIPT_DIR=$(dirname "$0")
cd SCRIPT_DIR
source ../../../venv/bin/activate

# Process and plot the data
python process.py
python plot.py
python calc_stats.py


