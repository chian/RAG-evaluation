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
        encoded_input = self._tokenizer(page_contents, padding=True, truncation=True, return_tensors='pt', max_length=512)

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
        encoded_input = self._tokenizer(input_data, padding=True, truncation=True, return_tensors='pt', max_length=512)

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

# Create the parser
parser = argparse.ArgumentParser(description="--input_file --test_file --model_name")

# Add arguments
parser.add_argument('--input_dir', type=str)
parser.add_argument('--test_file', type=str)
parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf")

args = parser.parse_args()

# Load the tokenizer and model from HuggingFace
model_name = args.model_name
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name)

# Check if the tokenizer has a pad token; if not, set it
#if tokenizer.pad_token is None:
#    tokenizer.pad_token = tokenizer.eos_token
    
# Load input files from input_dir
loader = DirectoryLoader(args.input_dir, glob="*", loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=10)
doc_chunks = text_splitter.split_documents(documents)
#print(doc_chunks)
#texts = [doc.page_content for doc in doc_chunks]
#print(type(texts))
#print(type(texts[0]))
#print(texts[0])

# Example usage
embedding_function = LocalEmbeddingFunction(model_name)
#embeddings = embedding_function(texts)
#print(embeddings)

vectordb = Chroma.from_documents(documents=doc_chunks, embedding=embedding_function, persist_directory="chroma_db")

# Test retriever
retriever = vectordb.as_retriever()
docs = retriever.get_relevant_documents( "What is the name of cpd32011?" )
print("RELEVANT DOCS")
print(docs)
