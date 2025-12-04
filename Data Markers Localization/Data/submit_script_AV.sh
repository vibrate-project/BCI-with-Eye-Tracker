#!/bin/bash

# Run this as "qsub submit_script.sh"

#PBS -l nodes=1:ppn=16          # Number of nodes
#PBS -l walltime=48:00:00       # Duration
#PBS -N script                  # Name job
#PBS -q compphys                # Queue

#PBS -M "teodor.vakarelsky@gmail.com"
#PBS -m e

cd $PBS_O_WORKDIR

cd ..
time python3 MainSynthetic.py

