from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import argparse
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
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

class LocalEmbeddingFunction(EmbeddingFunction[Documents]):
    def __init__(
            self,
            model_name: str,
            normalize_embeddings: bool = True,
    ):
        """
        Initializes the embedding function with a specified model from HuggingFace.

        Parameters:
        - model_name (str): The name of the model to load from HuggingFace.
        """
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
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
        embeddings = model_output.last_hidden_state.mean(dim=1).tolist()
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
        embeddings = model_output.last_hidden_state.mean(dim=1).tolist()
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

#Loading or making Chroma database
embedding_function = LocalEmbeddingFunction(model_name)
if (os.path.exists(args.persist_dir) and os.listdir(args.persist_dir)) and args.revector==False:
    print("Loading existing ChromaDB from", args.persist_dir)
    vectordb = Chroma(embedding_function=embedding_function,persist_directory=args.persist_dir)
else:
    print("Creating new ChromaDB at", args.persist_dir)
    # Load input files from input_dir
    loader = DirectoryLoader(args.input_dir, glob="*", loader_cls=TextLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)
    doc_chunks = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(documents=doc_chunks, embedding=embedding_function, persist_directory=args.persist_dir)

# Initialize retriever
zone = 1791
retriever = vectordb.as_retriever(search_kwargs={"k": zone})
#docs = retriever.get_relevant_documents("What is the name of cpd32028?")
#print("RELEVANT DOCS",len(docs))
#print(docs)

#Run VectorDB test: finding relevant documents - on base model vectors
test_filename = args.test_file
test_results = VectorTest(test_filename=test_filename, retriever=retriever)
print(test_results)


