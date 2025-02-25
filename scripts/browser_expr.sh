#!/bin/bash
 
#SBATCH --account=le-lab
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=0:01:00
#SBATCH --partition=general
#SBATCH --job-name=smoLM_test
#SBATCH --output=smoLM_test-%j.out
#SBATCH --error=smoLM_test-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sj110@iu.edu


 
module load anaconda/3

conda activate browser-gym
srun --x11 --pty bash python3 -m run_demo --start_url https://sj110.pages.iu.edu
