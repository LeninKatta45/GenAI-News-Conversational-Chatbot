import os
import time
import dill
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = "AIzaSyDLq67oYC-Vw60TU0vzszeDcOf3C6X7xG8"
os.environ["GOOGLE_API_KEY"] = api_key
loader=[]
app = FastAPI()

class URLInput(BaseModel):
    urls: list[str]

class QueryInput(BaseModel):
    query: str

class SummarizeInput(BaseModel):
    url_index: int

file_path = "faiss_store_openai.pkl"

def url_access(urls):
    global loader
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data

def chunk_access(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data)
    return docs

def split_combine_singleUrl(data, index):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(data[index:index+1])
    return docs

prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
```{text}```
BULLET POINT SUMMARY:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

llm = ChatGoogleGenerativeAI(
    google_api_key=os.environ["GOOGLE_API_KEY"],
    model="gemini-pro",
    temperature=0.6,
    convert_system_message_to_human=True,
)

stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

@app.post("/process_urls/")
async def process_urls(input: URLInput):
    urls = input.urls
    data = url_access(urls)
    docs = chunk_access(data)
    
    embeddings = GooglePalmEmbeddings()
    vectorstore_palm = FAISS.from_documents(docs, embeddings)
    
    with open(file_path, "wb") as f:
        dill.dump(vectorstore_palm, f)
    
    return {"message": "URLs processed and embeddings created successfully."}

@app.post("/query/")
async def query(input: QueryInput):
    query = input.query
    
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = dill.load(f)
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="map_rerank",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True,
            )
            result = chain({"query": query}, return_only_outputs=True)
            answer = result["result"]
            sources = {doc.metadata["source"] for doc in result["source_documents"][:3]}
            
            return {"answer": answer, "sources": list(sources)}
    else:
        return {"error": "Vector store not found. Please process URLs first."}


@app.post("/summarize/")
async def summarize(input: SummarizeInput):

    loader_new = UnstructuredURLLoader(urls=[loader.urls[input.url_index]])
    datas = loader_new.load()
    doc=split_combine_singleUrl(datas,input.url_index)
    try:
        summary = stuff_chain.run(doc)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}
   
