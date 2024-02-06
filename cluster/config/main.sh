# This file holds all the environment variables for the cluster that constitute the job configuration.

# if number of arguments is different from 2 including the script name, print error message and exit
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo "Usage: source cluster/config/main.sh <environment>_<seed>"
    exit 1
fi

# The following variables are used by the sbatch scripts
# The name of the job
export SBATCH_JOB_NAME="$1"

# The maximum time the job can run for
export SBATCH_TIMELIMIT="100:00:00"
