import os

from typing import (
    List,
    Union,
    Sequence,
)


# --------------------- Vector Stores  ------------------------- #
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.api import AdminAPI, AsyncClientAPI, ClientAPI
from chromadb.api.models.Collection import Collection


# -------------------------- Constants --------------------- #

PROJECT_FOLDER = "/Users/eric/DocumentsLocal/GitRepos/Dev/hackathonprep"
DATA_FOLDER_PATH =  os.path.join(PROJECT_FOLDER, "data")
DATA_DB_FOLDER_PATH = os.path.join(PROJECT_FOLDER, "data_db")


# -------------------------- Chroma ------------------------- #

def get_chroma_client() -> ClientAPI:
    vector_store_path = os.path.join(DATA_DB_FOLDER_PATH, "chroma.db")

    return chromadb.PersistentClient(
        path=vector_store_path,
    )


def get_chroma_vector_store() -> ChromaVectorStore:
    # initialize client, setting path to save data
    chroma_client = get_chroma_client()

    chroma_collection = chroma_client.get_or_create_collection(
        name="llamacollection",
    )   

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    return vector_store
