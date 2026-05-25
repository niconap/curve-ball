# Curve-ball

## Environment installation (on an HPC)

This code was tested with PyTorch 2.0.1, cuda 11.8 and torch\_geometrics 2.3.1 and is based on the [GGBall](https://github.com/AI4Science-WestlakeU/GGBall) repository

Download anaconda/miniconda if needed and optionally install mamba. If you do not want to install mamba, edit `jobs/install_env.job` to install using `conda` instead.

Use the `jobs/install_env.job` file to install the environment. It might be necessary to change the `module load` commands to match with the HPC you are using.

## Run the code

Many job files are provided in the `jobs` directory. Each job file runs a different experiment. See the name of each job file to determine which experiment it is.

### Example: training for `comm20` for several curvatures

To perform a full training pipeline for the `comm20` dataset, run jobs in the following order:
1. Start by running `comm20_array.job`, which starts five processes for training a HVQVAE, where each process uses a different curvature.
2. The previous step should have generated checkpoints, which we can use to run `comm20_hyp_flow_array.job`. Make sure to edit this job file to match the paths to the checkpoints to your system and be careful to match the right checkpoint to the right curvature. This too will generate five processes, each using a different curvature.
