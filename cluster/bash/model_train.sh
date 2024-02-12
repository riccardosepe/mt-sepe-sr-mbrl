#!/bin/bash

# if number of cmd arguments is different from 2, print error message and exit
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: bash model_train.sh <environment> <seed>"
    exit 1
fi

# first argument must be among the following: acrobot, acro3bot, cartpole, cart2pole, cart3pole, pendulum, reacher
case $1 in
    "acrobot" | "acro3bot" | "cartpole"  | "cart2pole" | "cart3pole" | "pendulum" | "reacher" | "soft_reacher")
        environment=$1;;
    *)
        echo "Invalid environment"
        echo "Usage: bash model_train.sh <environment> <seed>"
        exit
        ;;
esac


# second argument must be an integer
if ! [[ $2 =~ ^[0-9]+$ ]]; then
    echo "Error: <seed> must be an integer."
    echo "Usage: bash model_train.sh <environment> <seed>"
    exit 1
else
    seed=$2
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
export SBATCH_JOB_NAME=model_train_"$seed"

# The maximum time the job can run for
export SBATCH_TIMELIMIT="60:00:00"

sbatch  cluster/sbatch/$cluster.sbatch train_model.py --seed "$seed"
