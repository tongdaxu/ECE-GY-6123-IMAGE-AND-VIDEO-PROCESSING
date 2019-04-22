#!/bin/bash

#SBATCH --job-name=jupyter
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1

module purge
module load python3/intel/3.6.3
module load cuda/9.0.176
module load cudnn/9.0v7.0.5
module load anaconda3/4.3.1
source activate cs231

port=$(shuf -i 6000-9999 -n 1)

/usr/bin/ssh -N -f -R $port:localhost:$port log-0
/usr/bin/ssh -N -f -R $port:localhost:$port log-1

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

jupyter notebook --no-browser --port $port --notebook-dir=$(pwd)