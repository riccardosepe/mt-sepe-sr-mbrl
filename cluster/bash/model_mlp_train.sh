#!/bin/bash

# if number of cmd arguments is different from 1 or 2, print error message and exit
if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: bash model_train.sh <seed> [learning_rate]"
    exit 1
fi

if [ "$2" == "" ]; then
    learning_rate_param=""
else
    learning_rate_param="--lr $2"
fi

# second argument must be an integer
if ! [[ $1 =~ ^[0-9]+$ ]]; then
    echo "Error: <seed> must be an integer."
    echo "Usage: bash model_train.sh <seed> [learning_rate]"
    exit 1
else
    seed=$1
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
export SBATCH_JOB_NAME=model_mlp_train_"$seed"

# The maximum time the job can run for
export SBATCH_TIMELIMIT="15:00:00"

sbatch  cluster/sbatch/$cluster.sbatch train_model_mlp.py --seed "$seed" $learning_rate_param


