
# 5 hour  job
#$ -l h_rt=5:0:0
# Use current directory and environment
#$ -cwd -V
# A single P100 card, 6 CPU cores and 64GB system memory
# (one quarter of the available resource on a P100 node)

#$ -l coproc_v100=1

#Get email at start and end of the job
#$ -m be
module load anaconda

source activate cv4e2
# conda activate cv4e2

module unload cuda
# module add cuda/11.1.1, try a different cuda
module load cuda/10.1.168

nvidia-smi -L

# Run the Python code
python ct_classifier/train_save_epoch.py --config configs/ep25_56sp_anone_lr1e-3_snone_orig.yaml > outputep25test.log