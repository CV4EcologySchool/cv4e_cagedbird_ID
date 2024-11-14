
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
module unload cuda
module add cuda/11.1.1

source activate cv4e2


nvidia-smi -L

# Run the Python code
python ct_classifier/train.py --config configs/ep100_56sp_anone_lr1e-3_snone_orig.yaml > outputep100.log