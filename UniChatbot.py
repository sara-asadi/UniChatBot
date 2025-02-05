from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from transformers import AutoTokenizer
from langchain.schema import Document
import nltk
nltk.download('punkt_tab')

import nltk
nltk.download('averaged_perceptron_tagger_eng')


import sys

import os
import glob
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader

# Define the folder path in Google Drive
folder_path = Path(__file__).parent.parent / "ProjectNLP_Data"

def file_loader(folder_path):
  # Find all PDFs and DOC/DOCX files
  pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
  doc_files = glob.glob(os.path.join(folder_path, "*.doc")) + glob.glob(os.path.join(folder_path, "*.docx"))

  # Load all PDFs
  pdf_docs = [PyPDFLoader(pdf) for pdf in pdf_files]

  # Load all DOC/DOCX files
  doc_docs = [UnstructuredFileLoader(doc) for doc in doc_files]

  # Combine all documents
  all_documents = pdf_docs + doc_docs
  return all_documents;

all_documents = file_loader(folder_path)

def split_text_by_tokens(text, model_name="bert-base-multilingual-cased", chunk_size=250, chunk_overlap=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokens = tokenizer.encode(text, add_special_tokens=False)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        decoded_chunk = tokenizer.decode(chunk).strip()
        if decoded_chunk:  # Ensure non-empty chunks
            chunks.append(Document(page_content=decoded_chunk))
        start += chunk_size - chunk_overlap  # Move start index forward, considering overlap

    return chunks

def ingest():
    # Load documents
    pages = []
    for loader in all_documents:
        pages.extend(loader.load_and_split())

    # Apply token-based splitting
    processed_chunks = []
    for doc in pages:
        processed_chunks.extend(split_text_by_tokens(doc.page_content))

    print(f"Split {len(pages)} documents into {len(processed_chunks)} token-based chunks.")

    # Embedding and vector store creation
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    Chroma.from_documents(documents=processed_chunks, embedding=embedding, persist_directory="./sql_chroma_db")

ingest()

def rag_chain():
    model = ChatOllama(model="llama3")
    #
    prompt = PromptTemplate.from_template(
        """
        <s> [Instructions] You are a friendly assistant. Answer the question in Persian based only on the following context.
        If you don't know the answer, then reply, No Context availabel for this question {input}. [/Instructions] </s>
        [Instructions] Question: {input}
        Context: {context}
        Answer: [/Instructions]
        """
    )
    #Load vector store
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vector_store = Chroma(persist_directory="./sql_chroma_db", embedding_function=embedding)

    #Create chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 3,
            "score_threshold": 0.5,
        },
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    chain = create_retrieval_chain(retriever, document_chain)
    #
    return chain

def ask(query: str):
    #
    chain = rag_chain()
    # invoke chain
    result = chain.invoke({"input": query})
    # print results
    print(result["answer"])
    for doc in result["context"]:
        print("Source: ", doc.metadata["source"])

ask("اعضای هیپت علمی کیا هستند؟")