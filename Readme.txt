# If encountering "Your shell has not been properly configured to use 'conda activate'.", then "conda deactivate" from (base)
source activate base	
conda deactivate
	
conda init bash
conda create --name conda39-madt python=3.9
conda activate conda39-madt
pip install -r requirements.txt -t ./pipenv
conda deactivate
conda clean --all	# Purge cache and unused apps
condo info

source madt.sh

## Pull additional repos
vcs import < madt_repos.txt

python3 -m main_atari
python3 -m main_gym

python3 -m main_atari --convlstm --mask_double --env BreakoutNoFrameskip-v4 --load-model BreakoutNoFrameskip-v4_Mask-A3C-double+ConvLSTM_best --num-episodes 100 --gpu-ids 0 --render
