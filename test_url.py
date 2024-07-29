from langchain_community.document_loaders import UnstructuredURLLoader
import pytest
from main import url_access, chunk_access
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import GooglePalmEmbeddings

t = []


def test_load_urls():
    d = [
        "https://timesofindia.indiatimes.com/sports/cricket/england-in-india/india-vs-england-kl-rahul-ruled-out-jasprit-bumrah-returns-for-5th-test/articleshow/108102429.cms"
    ]
    global t
    t = url_access(d)
    assert len(t) != 0


def test_load_chunk():
    assert len(chunk_access(t)) > 0
