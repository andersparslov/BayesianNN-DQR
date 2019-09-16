#!/bin/sh
## Request 1 core
#BSUB -n 1
#BSUB -R "span[hosts=1]"
## Request 15 minutes of walltime
#BSUB -W 00:15
## Request 500 MB of memory *per core*
#BSUB -R "rusage[mem=500MB]"

module load numpy/1.13.1-python-3.6.2-openblas-0.2.20

python3 test.py