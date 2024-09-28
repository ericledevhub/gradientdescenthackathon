import os

# --------------------- Vector Stores  ------------------------- #

from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from chromadb.api import AdminAPI, AsyncClientAPI, ClientAPI
from chromadb.api.models.Collection import Collection

# -------------------------- Chroma ------------------------- #

def get_chroma_client(
        project_path: str
    ) -> ClientAPI:

    vector_db_folder_path = os.path.join(project_path, "data_db")

    if not os.path.exists(vector_db_folder_path):
        print(f"Created folder {vector_db_folder_path} ...")
        os.makedirs(vector_db_folder_path)

    vector_store_path = os.path.join(vector_db_folder_path, "chroma.db")

    return chromadb.PersistentClient(
        path=vector_store_path,
    )


def get_chroma_vector_store(
        project_path: str
    ) -> ChromaVectorStore:

    # initialize client, setting path to save data
    chroma_client = get_chroma_client(
        project_path=project_path,
    )

    chroma_collection = chroma_client.get_or_create_collection(
        name="llamacollection",
    )   

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    return vector_store
