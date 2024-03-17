#!/bin/bash

# if number of cmd arguments is different from 2, print error message and exit
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: bash $0 <seed> <model-type>"
    exit 1
fi


# first argument must be an integer
if ! [[ $1 =~ ^[0-9]+$ ]]; then
    echo "Error: <seed> must be an integer."
    echo "Usage: bash $0 <seed> <model-type>"
    exit 1
else
    seed=$1
fi

# second argument must be either "mlp" or "lnn"
if [ "$2" != "mlp" ] && [ "$2" != "lnn" ]; then
    echo "Error: <model-type> must be either 'mlp' or 'lnn'."
    echo "Usage: bash $0 <seed> <model-type>"
    exit 1
else
    model_type=$2
fi


# if current folder is not called 'Physics_Informed_Model_Based_RL', print a warning
if [ "${PWD##*/}" != "Physics_Informed_Model_Based_RL" ]; then
    echo "WARNING: current folder is not called Physics_Informed_Model_Based_RL"
fi

if [ "$HOSTNAME" = "legionlogin" ]; then
    cluster="legion"
else
    cluster="delftblue"
fi

# The following variables are used by the sbatch scripts
# The name of the job
export SBATCH_JOB_NAME="sac_model_soft_reacher_$seed"

# The maximum time the job can run for
export SBATCH_TIMELIMIT="100:00:00"


sbatch  cluster/sbatch/$cluster.sbatch sac_with_model.py --env soft_reacher --mode train --episodes 500 --seed "$seed" --model-type "$model_type"

