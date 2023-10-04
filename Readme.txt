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

python3 -m main
