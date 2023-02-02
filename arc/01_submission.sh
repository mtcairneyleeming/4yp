#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=4yp_01
#SBATCH --time=06:00:00
#SBATCH --partition=short
#SBATCH -o ./arc/reports/slurm-%j.out # STDOUT
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=max.cairneyleeming@lmh.ox.ac.uk
#SBATCH --mem=64G



export WORKING_DIR=$DATA/4yp
export CONDA_PREFIX=/data/coml-hawkes/lady6235/jax_numpyro_env
export FILE_TO_RUN=01_code.py


module load Mamba # note we are not using Mamba to build the environment, we just need to load into it

# set the Anaconda environment, and activate it:
source activate $CONDA_PREFIX


# change to the temporary $SCRATCH directory, where we can create whatever files we want
cd $SCRATCH
mkdir output # create an output folder, which we will copy across to $DATA when done
mkdir code

# copy across only what we need:
cp -R $WORKING_DIR/reusable   	. # the code we've actually written
cp -R $WORKING_DIR/plotting   	. # the code we've actually written
cp -R $WORKING_DIR/$FILE_TO_RUN . # the code we've actually written


echo "Environment variables:"
printenv | grep ^SLURM_* # print all SLURM config (# of tasks, nodes, mem, gpus etc.)
echo "Files copied across:"
tree


python ./$FILE_TO_RUN


# copy the output directory back across to $DATA
mkdir $WORKING_DIR/arc/outputs/$SLURM_JOB_ID
echo "Outputs created: "
tree ./output
rsync -av ./output/* $WORKING_DIR/arc/outputs/$SLURM_JOB_ID
