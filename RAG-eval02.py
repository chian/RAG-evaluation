from transformers import AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedModel
import torch
import torch.nn.functional as F
import argparse
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
import json
from typing import List,Union
from chromadb.api.types import (
    Document,
    Documents,
    Embedding,
    Image,
    Images,
    EmbeddingFunction,
    Embeddings,
    is_image,
    is_document,
)
import re
#import pdb

class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            #model_name: str,
            tokenizer: PreTrainedTokenizer,
            model: PreTrainedModel,
            normalize_embeddings: bool = True,
    ):
        """
        Initializes the embedding function with a specified model from HuggingFace.
        """
        self._tokenizer = tokenizer
        self._model = model
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._normalize_embeddings = normalize_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        return cast(
            Embeddings,
            self._model.encode(
                list(input),
                convert_to_numpy=True,
		normalize_embeddings=self._normalize_embeddings,
                ).tolist()
            )

        self._normalize_embeddings = normalize_embeddings

    def embed_documents(self, documents: List[Documents]) -> List[List[float]]:
        
        # Extract page_content from each Document dictionary
        #print(documents)
        page_contents = [doc for doc in documents]
        #print(page_contents)
        #print(len(page_contents))
        
        # Tokenize the input documents
        encoded_input = self._tokenizer(page_contents, padding=True, truncation=True, return_tensors='pt', max_length=1024)

        # Generate embeddings using the model
        with torch.no_grad():
            model_output = self._model(**encoded_input)

        # Extract embeddings
        hidden_states = model_output.hidden_states
        last_layer_hidden_states = hidden_states[-1]
        embeddings = last_layer_hidden_states.mean(dim=1).tolist()
        #embeddings = model_output.last_hidden_state.mean(dim=1).tolist()
        #print(len(embeddings))
        
        return embeddings

    def embed_query(self, input_data: str) -> List[float]:
        #Converts a list of text documents into embeddings.

        #Parameters:
        #- documents (List[str]): A list of text documents to convert.

        #Returns:
        #- List[List[float]]: A list of embeddings, one per document.

        # Tokenize the input documents. This will turn each document into a format the model can understand.
        encoded_input = self._tokenizer(input_data, padding=True, truncation=True, return_tensors='pt', max_length=1024)

        # Generate embeddings using the model. We'll use the last hidden state for this purpose.
        with torch.no_grad():
            model_output = self._model(**encoded_input)

        # Extract embeddings from the model output. Depending on the model, you might want to adjust this.
        # For many models, taking the mean of the last hidden state across the token dimension is a good starting point.
        #pdb.set_trace()
        #print("model_output:",model_output)
        hidden_states = model_output.hidden_states
        last_layer_hidden_states = hidden_states[-1]
        embeddings = last_layer_hidden_states.mean(dim=1).tolist()
        #embeddings = model_output.last_hidden_state.mean(dim=1).tolist()
        #print(len(embeddings))
        #print(len(embeddings[0]))
        #print(embeddings[0])
        
        return embeddings[0]

def VectorTest(test_filename, retriever):
    super_index_list = []
    first_index_list = []
    print("query_keyword\tanswer_key\tsearch_rank")
    with open(test_filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line.startswith('#'):
                fields = line.split("\t")
                query = fields[0]
                answer_key = fields[1]
                docs = retriever.get_relevant_documents(query)
                pattern_temp = re.compile(r'(?:^|\W)' + re.escape(str(answer_key)) + r'(?:$|\W)')
                index = 1
                index_list = []
                for doc in docs:
                    #print(re.escape(str(doc.page_content).strip()))
                    if pattern_temp.search(str(doc.page_content).strip()):
                        if len(index_list)==0:
                            first_index_list.append(index)
                        index_list.append(index)
                        super_index_list.append(index)
                        #print("##",doc.page_content)
                    index += 1
                print(query, answer_key, index_list)
                #print("#",docs[0].page_content)
                if len(index_list) == 0:
                    super_index_list.append(zone)

    avg_all_rank = sum(super_index_list)/len(super_index_list)
    avg_first_rank = sum(first_index_list)/len(first_index_list)
    print("Average Search Rank:",avg_all_rank)
    print("Average First Hit Rank:",avg_first_rank)
    return (avg_all_rank, avg_first_rank)

# Create the parser
parser = argparse.ArgumentParser(description="--input_file --test_file --model_name")

# Add arguments
parser.add_argument('--input_dir', type=str)
parser.add_argument('--test_file', type=str)
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf")
parser.add_argument('--persist_dir', type=str, default="chroma_db")
parser.add_argument('--revector', action='store_true')

args = parser.parse_args()

# Load the tokenizer and model from HuggingFace
model_name = args.model_name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.config.use_cache = False
model.config.output_hidden_states = True

#Loading or making Chroma database
embedding_function = LocalEmbeddingFunction(tokenizer,model)
if (os.path.exists(args.persist_dir) and os.listdir(args.persist_dir)) and args.revector==False:
    print("Loading existing ChromaDB from", args.persist_dir)
    vectordb = Chroma(embedding_function=embedding_function,persist_directory=args.persist_dir)
else:
    print("Creating new ChromaDB at", args.persist_dir)
    # Load input files from input_dir
    loader = DirectoryLoader(args.input_dir, glob="*", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=doc_chunks, embedding=embedding_function, persist_directory=args.persist_dir)

# Initialize retriever
zone = 255 #1791
retriever = vectordb.as_retriever(search_kwargs={"k": zone})
#docs = retriever.get_relevant_documents("What is the name of cpd32028?")
#print("RELEVANT DOCS",len(docs))
#print(docs)

#Run VectorDB test: finding relevant documents - on base model vectors
test_filename = args.test_file
test_results = VectorTest(test_filename=test_filename, retriever=retriever)
#print(test_results)

#Fine-tune model
from datasets import Dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset
from transformers import TrainingArguments, pipeline
from trl import SFTTrainer

def get_text_data(filename):
    instruction = "Learn this biology information. "   # same for every line here                                                                               
    list_of_text_dicts = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            text = ("### Instruction: \n" + instruction + "\n" +
                    "### Input: \n" + line + "\n" +
                    "### Response :\n" + line)
            list_of_text_dicts.append( { "text": text } )
    return list_of_text_dicts

def user_prompt(human_prompt):
    # must chg if dataset isn't formatted as Alpaca
    prompt_template=f"### HUMAN:\n{human_prompt}\n\n### RESPONSE:\n"
    return prompt_template

base_model_name = model_name

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    #load_in_8bit=True,
    #load_in_4bit=True,
    #quantization_config=bnb_config,
    device_map='auto',
)
base_model.config.use_cache = False
print(base_model)

footprint = base_model.get_memory_footprint()
print("BASE MEM FOOTPRINT",footprint)

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.pad_token = tokenizer.eos_token

pipe = pipeline(
    task="text-generation",
    model=base_model,
    tokenizer=tokenizer,
    max_length=150,
    repetition_penalty=1.15,
    top_p=0.95
)

prompt = f"What is the name of the compound with ModelSEED ID cpd0007?"
result = pipe(user_prompt(prompt))
print(result[0]['generated_text'])

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    # target modules varies from model to model
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(base_model, lora_config)

filename = os.path.join(args.input_dir, "compounds_top100.txt")
print(filename)
data = get_text_data(filename)
train_dataset = Dataset.from_dict({key: [dic[key] for dic in data] for key in data[0]})
print(train_dataset)
print(train_dataset[0])
output_dir = os.path.join(args.input_dir,"TMP_RESULTS")

training_arguments = TrainingArguments(
    output_dir = output_dir,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    save_strategy="epoch",
    logging_steps=10,
    num_train_epochs=10,
    max_steps=80,
    learning_rate=2e-4,
    fp16=False, #False on a mac
    # max_grad_norm = 0.3,
    # max_steps = 300,
    warmup_ratio = 0.03,
    group_by_length=True,
    lr_scheduler_type = "linear",  # vs constant
    report_to = "none",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    dataset_text_field="text",
    args=training_arguments,
    max_seq_length=1024,
)

trainer.train()

adaptor_filename = os.path.join(args.input_dir,"TMP_adapter")
trainer.save_model(adaptor_filename)
adapter_model = model
print("Lora Adapter saved")

# can't merge the 8 bit/4 bit model with Lora so reload it
repo_id = args.model_name
use_ram_optimized_load=False

base_model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    # trust_remote_code=True,
    device_map='auto',
)
base_model.config.use_cache = False
base_model.config.output_hidden_states = True

footprint = base_model.get_memory_footprint()
print("BASE MEM FOOTPRINT",footprint)

# Load Lora adapter                     
peft_model = PeftModel.from_pretrained(
    base_model,
    adaptor_filename,
)
peft_model.config.use_cache = False
peft_model.config.output_hidden_states = True

#Run VectorDB test: finding relevant documents - on fine-tuned model vectors

#Loading or making Chroma database
embedding_function = LocalEmbeddingFunction(tokenizer,peft_model)
second_chroma_dir = os.path.join(args.input_dir,args.persist_dir)
if (os.path.exists(second_chroma_dir) and os.listdir(second_chroma_dir)) and args.revector==False:
    print("Loading existing ChromaDB from", second_chroma_dir)
    vectordb = Chroma(embedding_function=embedding_function,persist_directory=second_chroma_dir)
else:
    print("Creating new ChromaDB at", second_chroma_dir)
    # Load input files from input_dir
    loader = DirectoryLoader(args.input_dir, glob="*", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(documents)
    print("documents chunked. moving on to vectorization...")
    vectordb = Chroma.from_documents(documents=doc_chunks, embedding=embedding_function, persist_directory=second_chroma_dir)
    print("vectorization complete. moving on to VectorTest...")

# Initialize retriever
#zone = 109 #1791
retriever = vectordb.as_retriever(search_kwargs={"k": zone})
#Run VectorDB test: finding relevant documents - on base model vectors
test_filename = args.test_file
test_results = VectorTest(test_filename=test_filename, retriever=retriever)
#print(test_results)
