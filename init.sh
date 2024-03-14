export HF_DATASETS_CACHE=/lus/eagle/projects/argonne_tpc/chia-llama2/.cache
export TRANSFORMERS_CACHE=/lus/eagle/projects/argonne_tpc/chia-llama2/.cache
export HF_HOME=~/.huggingface

module load conda/2023-10-04
https_proxy=http://proxy.alcf.anl.gov:3128
http_proxy=http://proxy.alcf.anl.gov:3128
conda activate /lus/eagle/projects/argonne_tpc/chia-llama2/conda_envs/autotrain
#source /lus/eagle/projects/argonne_tpc/chia-llama2/virtualenv_envs/autotrain/bin/activate
#venv directories that some packages such as autotrain need
#export PYTHONPATH=/lus/eagle/projects/argonne_tpc/chia-llama2/virtualenv_envs/autotrain/lib/python3.11/site-packages
#export LD_LIBRARY_PATH=/lus/eagle/projects/argonne_tpc/chia-llama2/virtual_env/autotrain2/lib64:$LD_LIBRARY_PATH
#export PATH=/lus/eagle/projects/argonne_tpc/chia-llama2/virtual_env/autotrain2/bin:$PATH
#export CUDA_HOME=/lus/eagle/projects/argonne_tpc/chia-llama2/virtual_env/autotrain2/lib/python3.11/site-packages

#export CUDA_VISIBLE_DEVICES=0,1,2,3
#export CUDA_VISIBLE_DEVICES=0 #head node testing
