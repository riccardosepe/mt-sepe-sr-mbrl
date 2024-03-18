#!/bin/bash

# Per-cluster variables

#SBATCH --partition=gpu
#SBATCH --account=innovation
#SBATCH --mail-type=ALL
#SBATCH --mail-user=riccardo.sepe@studenti.polito.it
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Load python
module load 2022r2
module load python/3.8.12

# Load nvidia module
module load cuda/12.1


# Run the script
python "$@"