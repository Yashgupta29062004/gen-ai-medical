from pinecone import Pinecone
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

#extract text from pdf files
def load_pdf_files(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents=loader.load()
    return documents

extracted_data=load_pdf_files("data")



from typing import List
from langchain.schema import Document
def filter_to_minimal_docs(docs:List[Document])->List[Document]:
    """given a list of document obeject, return a new list of document object that only contain 
    source data by removing the meta data"""
    minimal_docs:List[Document]=[]
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src}
            )
        )
    return minimal_docs
minimal_docs=filter_to_minimal_docs(extracted_data)

    

#split the document int small chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        

    )
    texts_chunk=text_splitter.split_documents(minimal_docs)
    return texts_chunk

text_chunks=text_split(minimal_docs)
print(f"number of chunks:{len(text_chunks)}")

from langchain.embeddings import HuggingFaceBgeEmbeddings
def download_embedding():
    """downloadand return the hugging face embeeding model"""
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embedding=HuggingFaceBgeEmbeddings(
        model_name=model_name   
    )
    return embedding
embedding=download_embedding()


from dotenv import load_dotenv
import os
load_dotenv()
PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GEMINI_API_KEY"]=GEMINI_API_KEY

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")


os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["GEMINI_API_KEY"]=GEMINI_API_KEY

from pinecone import ServerlessSpec
pc = Pinecone(api_key="pcsk_39nRch_SYJ6YgGhHmcJwRAKuP4R6EV82d7ZQi8SdtU6vMF9way8i3zvx8aYLshRyFoPRzJ")
index_name="medibot"
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    for name in existing_indexes:
        print(f"Deleting old index: {name}...")
        pc.delete_index(name)
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws",region="us-east-1" )
    )
index=pc.Index(index_name)

from langchain_pinecone import PineconeVectorStore
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name

)



