#!/bin/bash
  
#SBATCH --account=envi.p_invert          # Your account 
#SBATCH --qos=48h
#SBATCH --time=48:00:00
#SBATCH -p mpp

#SBATCH -N 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread

#SBATCH --output=final_out_%x.%j



module purge
module load    miniconda3/
module load    python/3.10.4
source /albedo/work/projects/p_inverT/envts/emcee/bin/activate

cd /albedo/work/projects/p_inverT

# Run your script as a module
#srun --cpu-bind=none python -m Ice_BT.code.Realistic_smooth_signal_PA1_28_1mK
#srun --cpu-bind=none python -m Ice_BT.code.Realistic_smooth_signal_PA1_28_1mK_conti1
#srun --cpu-bind=none python -m Ice_BT.code.Realistic_smooth_signal_PA1_28_1mK_conti2


srun --cpu-bind=none python -m Ice_BT.code.Analysis_Realistic_smooth_signal_PA1_28_1mK
