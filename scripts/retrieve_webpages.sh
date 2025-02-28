#!/bin/bash

#SBATCH -J dataset1
#SBATCH --gres gpu:L40S:1
#SBATCH -o %j.txt
#SBATCH -e %j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=username@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH -A le-lab

#Load any modules that your program needs
module load anaconda3
conda init
conda activate browser-gym

#Run your program
python3 -m src.actions.make_dataset get_webs