#!/bin/bash

#SBATCH -n 20                              # Number of cores
#SBATCH --time=24:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=abm
#SBATCH --output=abm.out
#SBATCH --error=abm.err

source euler_scripts/setup.sh

python main.py
    