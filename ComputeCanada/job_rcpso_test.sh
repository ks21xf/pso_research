#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-bmombuki
#SBATCH --mem=8G
cd ..
module load python/3.10

source ve/bin/activate #remove if not using a VE

python RCPSO_test.py
