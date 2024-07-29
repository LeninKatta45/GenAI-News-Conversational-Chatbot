import os
import streamlit as st
import time
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
import dill
from langchain_google_genai import ChatGoogleGenerativeAI

api_key = "AIzaSyDLq67oYC-Vw60TU0vzszeDcOf3C6X7xG8"
os.environ["GOOGLE_API_KEY"] = api_key


def url_access(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    return data


def chunk_access(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    return docs


st.title(r"QuantumChat:An Ai News Research Tool 🤖 ")
st.sidebar.title("News Article URLs")


llm = ChatGoogleGenerativeAI(
    google_api_key=os.environ["GOOGLE_API_KEY"],
    model="gemini-pro",
    temperature=0.6,
    convert_system_message_to_human=True,
)
urls = []
summarize_buttons = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    summarize_buttons.append(st.sidebar.button(f"Summarize-{i+1}"))
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"
mp = st.empty()
main_placeholder = st.empty()

# llm=GooglePalm(google_api_key=api_key,temperature=0.4,max_output_tokens=512)

if process_url_clicked:
    # load data
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = url_access(urls)

    # split data
    docs = chunk_access(data)
    # create embeddings and save it to FAISS index
    embeddings = GooglePalmEmbeddings()
    vectorstore_palm = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        dill.dump(vectorstore_palm, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = dill.load(f)
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="map_rerank",
                retriever=vectorstore.as_retriever(),
                return_source_documents=True,
            )
            result1 = chain({"query": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result1["result"])

            # Display sources, if available
            if result1["source_documents"]:
                st.subheader("Sources:")
                se = set()
                i = 1
                for me in result1["source_documents"]:
                    if i == 3:
                        break

                    se.add(me.metadata["source"])
                    i = i + 1
                for s in se:
                    st.write(s)


# "****************************** S U M M A R Y *****************************"
def split_combine_singleUrl(j, data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    # As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
    docs = text_splitter.split_documents(data[j : j + 1])
    return docs


prompt_template = """Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text.
              ```{text}```
              BULLET POINT SUMMARY:
  """

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)


def summary(doc):
    try:

        st.header("Summary")
        st.write(stuff_chain.run(doc))
    except Exception as e:
        # st.header("Summary")
        st.write("Huge Context to Retain")


def load():
    loader = UnstructuredURLLoader(urls=urls)
    mp.text("Data Loading...Started...✅✅✅")
    st.empty()
    datas = loader.load()
    return datas


if summarize_buttons[0]:
    st.empty()
    j = 0
    data = load()
    doc = split_combine_singleUrl(j, data)
    summary(doc)

if summarize_buttons[1]:
    st.empty()
    j = 1
    data = load()
    doc = split_combine_singleUrl(j, data)
    summary(doc)

if summarize_buttons[2]:
    st.empty()
    j = 2
    data = load()
    doc = split_combine_singleUrl(j, data)
    summary(doc)
