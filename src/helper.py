
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
from langchain.schema import Document

# extract data from the pdf file
def load_pdf_file(data):
    loader=DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader   # type: ignore
    )
    documents=loader.load()
    return documents 


def filter_to_minimal_docs(docs:List[Document]) ->List[Document]:
    minimal_documents:List[Document]=[]
    for doc in docs:
        minimal_documents.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":doc.metadata.get("source")}
            )
        )
    return minimal_documents

# split the documents into smaller chunks
def text_splitter(minimal_docs):
    text_splitt=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,

    )
    texts=text_splitt.split_documents(minimal_docs)
    return texts

# downloading embedded model
from langchain_huggingface import HuggingFaceEmbeddings
def downloading_embedding():
    "download and return the Huggingface embedding model"
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    embeddings=HuggingFaceEmbeddings(
        model_name=model_name     
    )
    return embeddings


