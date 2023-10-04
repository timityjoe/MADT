#!/bin/bash

# See https://docs.omniverse.nvidia.com/isaacsim/latest/manual_standalone_python.html
echo "Setting up MADT Environment..."
source activate base	
conda deactivate
conda activate conda39-madt
echo "$PYTHON_PATH"
