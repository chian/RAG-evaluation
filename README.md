# RAG-evaluation
Train LLM against data and measure changes in retreiver performance

For any serious fine-tuning, the users of this will want to use a HPC cluster such as ALCF's Polaris

Polaris Instructions:
module load conda/2023-10-04
conda activate base
git clone [this-github]
pip install -r requirements.txt
python RAG-eval02.py --input_dir data_small --test_file tests_small/name2cpdID.txt --model_name ./dir_to/Llama-2-7b-chat-hf/
