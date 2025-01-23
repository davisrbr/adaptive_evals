# Create new environment with Python
conda create -n inspect4 python=3.10

conda init bash
# Activate the new environment
conda activate inspect4

# Install PyTorch CPU version
conda install pytorch cpuonly -c pytorch

# Install inspect-ai via pip
pip install inspect-ai
pip install openai anthropic

# Install other required packages via conda
conda install -c huggingface transformers
conda install -c conda-forge sentence-transformers datasets
conda install tqdm