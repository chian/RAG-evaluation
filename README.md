# RAG-evaluation

This project aims to train Large Language Models (LLM) against a dataset and measure changes in retriever performance. It is designed for researchers and practitioners who are interested in fine-tuning LLMs for improved retrieval capabilities.

## Overview

The RAG-evaluation toolkit provides a structured approach to train LLMs and evaluate their retrieval performance across different datasets. This toolkit is particularly useful for those looking to conduct serious fine-tuning experiments.

## Prerequisites

For extensive fine-tuning tasks, users are encouraged to utilize a High-Performance Computing (HPC) cluster. An example of such a cluster is the Argonne Leadership Computing Facility's (ALCF) Polaris.

### Using Polaris

To get started with using the Polaris HPC cluster for your fine-tuning tasks, follow these instructions:

1. **Load Conda Environment**

   ```bash
   module load conda/2023-10-04
   conda activate base
   ```

2. **Copy Conda Environment Locally**

   ```bash
   conda create --clone base --prefix /path/to/envs/base-clone
   conda activate /path/to/envs/base-clone
   ```
   
3. **Pip install requirements**
   
   ```bash
   pip install --ignore-installed -r requirements.txt
   ```

6. **Activate you conda env**

   ```bash
   conda activate /path/to/your/directory/envname
   ```
   
7. **Clone the GitHub Repository**

   Replace `[this-github]` with the actual URL of the GitHub repository you intend to use.
   You may want to use a project directory for more storage space.

   ```bash
   git clone [this-github]
   ```

8. **Install Required Packages**

   Navigate to the cloned repository directory and install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

9. **Run the Evaluation Script**

   Execute the evaluation script with the necessary parameters:

   ```bash
   python RAG-eval02.py --input_dir data_small --test_file tests_small/name2cpdID.txt --model_name ./dir_to/Llama-2-7b-chat-hf/
   ```

## Contributing

We welcome contributions to improve the RAG-evaluation toolkit. Please feel free to submit pull requests or open issues to discuss potential features or bugs.

## License

Please include a section on licensing, specifying how others can use or contribute to your project.
