#!/bin/bash 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 180G
#SBATCH --time 50:00:00
#SBATCH --gres gpu:volta:2

#SBATCH --account vita

module load gcc/8.4.0-cuda  python/3.7.7
source /work/vita/sadegh/env/bin/activate
python -V
python train.py --config-module configs.deepsvg.hierarchical_ordered --data-type lyft --modes 3
