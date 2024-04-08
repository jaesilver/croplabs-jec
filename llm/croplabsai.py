import os 
import getpass 
from langchain import hub
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Ollama  
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()

class Croplabsai:
    #Define the repo ID 
    repo_id = "google/gemma-2b-it"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    # load the data
    loader = WebBaseLoader(
    web_path=("https://jaesilver.github.io/firstbookoffarming/"),
    )
    docs= loader.load()

    # splitting
    text_splitter= RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200,
    add_start_index = True,
    )
    all_splits = text_splitter.split_documents(docs)

    # create embeddings
    embedding = OllamaEmbeddings(
    model="nomic-embed-text"
    )

    # storing to chroma
    vectorstore = Chroma.from_documents(
    documents = all_splits,
    embedding = embedding,
    )

    #retrieving from chroma
    retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":6}
    )

    
    # retrival and generation
    prompt= hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
