#Load the pdf 
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

#filter to minimal
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

    

#split the document int small chunks
def text_split(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
        

    )
    texts_chunk=text_splitter.split_documents(minimal_docs)
    return texts_chunk


from langchain.embeddings import HuggingFaceBgeEmbeddings

def download_embedding():
    """downloadand return the hugging face embeeding model"""
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embedding=HuggingFaceBgeEmbeddings(
        model_name=model_name   
    )
    return embedding
embedding=download_embedding()