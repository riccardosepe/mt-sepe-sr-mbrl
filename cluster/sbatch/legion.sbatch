#!/bin/bash

# Per-cluster variables
#SBATCH --partition=cuda
#SBATCH --mail-type=ALL
#SBATCH --mail-user=riccardo.sepe@studenti.polito.it
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


# Load cuda
module load nvidia/cudasdk/11.6

# if no conda environment is currently active, first source ~/.bashrc, then activate 'legion_venv'
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    source ~/.bashrc
    conda activate legion_venv
fi

# Run the script
python "$@"