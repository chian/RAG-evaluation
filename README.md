# RAG-evaluation

This project aims to train Large Language Models (LLM) against a dataset and measure changes in retriever performance. It is designed for researchers and practitioners who are interested in fine-tuning LLMs for improved retrieval capabilities.

## Overview

The RAG-evaluation toolkit provides a structured approach to train LLMs and evaluate their retrieval performance across different datasets. This toolkit is particularly useful for those looking to conduct serious fine-tuning experiments.

## Prerequisites

For extensive fine-tuning tasks, users are encouraged to utilize a High-Performance Computing (HPC) cluster. An example of such a cluster is the Argonne Leadership Computing Facility's (ALCF) Polaris.

### Using Polaris

To get started with using the Polaris HPC cluster for your fine-tuning tasks, follow these instructions:

```bash
#!/bin/bash -l                                                                                                                                                                
#PBS -l select=1:system=polaris                                                                                                                                               
#PBS -l place=scatter                                                                                                                                                         
#PBS -l walltime=2:00:00                                                                                                                                                      
#PBS -l filesystems=home:eagle                                                                                                                                                
#PBS -j oe                                                                                                                                                                    
#PBS -q preemptable                                                                                                                                                           
#PBS -A argonne_tpc                                                                                                                                                           

# Configuration ------------------------                                                                                                                                      
ROOT=/lus/eagle/projects/argonne_tpc/chia-llama2/RAG-evaluation
DATA_DIR="$ROOT/data_small"

# ------------------                                                                                                                                                          
JOB_ID=$PBS_JOBID # Keep track of which job this is                                                                                                                           

# -------------------                                                                                                                                                         
# Instructions on multi-node PyTorch on Polaris                                                                                                                               
# https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/                                                                                                
#                                                                                                                                                                             
# Below settings are taken from there:                                                                                                                                        

# Enable the use of CollNet plugin.                                                                                                                                           
# CollNet allows for "in-network reductions" - apparently                                                                                                                     
# the switches themselves can do computation as the data is being shared (???)                                                                                                
export NCCL_COLLNET_ENABLE=1

# Use GPU Direct RDMA when GPU and NIC are on the same NUMA node.                                                                                                             
# Traffic will go through the CPU.                                                                                                                                            
export NCCL_NET_GDR_LEVEL=PHB

# ----------------                                                                                                                                                            

# Enable GPU-MPI (if supported by application)
export MPICH_GPU_SUPPORT_ENABLED=1


# Information on multi node setup                                                                                                                                             
export MASTER_ADDR=`head -n 1 $PBS_NODEFILE` # The first node in the list is the master node                                                                                  
export MASTER_PORT=29400 # Default pytorch port                                                                                                                               
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=1 #$(nvidia-smi -L | wc -l)                                                                                                                                   
NDEPTH=8
NTHREADS=1 # We use torch.distributed to spawn child processes                                                                                                                

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))
echo "NUM_OF_NODES= ${NNODES} TOTAL_NUM_RANKS= ${NTOTRANKS} RANKS_PER_NODE= ${NRANKS_PER_NODE} THREADS_PER_RANK= ${NTHREADS}"

# Set up MMAE environment                                                                                                                                                     
source "$ROOT/init.sh"
#export SOURCE_DATASET_NAME=TCGA                                                                                                                                              

echo "Executing in $PBS_O_WORKDIR/.."
cd "$PBS_O_WORKDIR"
echo `pwd`

python RAG-eval03.py --input_dir data_medium2 --test_file tests_small/name2cpdID.txt --model_name ../azton/Llama-2-7b-chat-hf
```

## Contributing

We welcome contributions to improve the RAG-evaluation toolkit. Please feel free to submit pull requests or open issues to discuss potential features or bugs.

## License

Please include a section on licensing, specifying how others can use or contribute to your project.
