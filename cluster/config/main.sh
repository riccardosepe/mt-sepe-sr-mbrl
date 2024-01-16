# This file holds all the environment variables for the cluster that constitute the job configuration.

# The following variables are used by the sbatch scripts
# The name of the job
export SBATCH_JOB_NAME="model_test"

# The maximum time the job can run for
export SBATCH_TIMELIMIT="100:00:00"
