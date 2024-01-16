#!/bin/bash

# if current folder is not called 'Physics_Informed_Model_Based_RL', print a warning
if [ "${PWD##*/}" != "Physics_Informed_Model_Based_RL" ]; then
    echo "WARNING: current folder is not called Physics_Informed_Model_Based_RL"
fi

if [ "$HOSTNAME" = "legionlogin" ]; then
    cluster="legion"
else
    cluster="delftblue"
fi

source cluster/config/main.sh

sbatch  cluster/sbatch/$cluster.sbatch mblr.py --env jax_pendulum --mode train --episodes 500 --seed 27 --eval-over 10
