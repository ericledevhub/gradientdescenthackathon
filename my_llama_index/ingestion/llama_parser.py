import os
from llama_parse import LlamaParse

from typing import (
    List,
    Union,
    Sequence,
)

# --------------------- Core  ------------------------- #
from llama_index.core.query_engine import CitationQueryEngine


# --------------------- Core  ------------------------- #
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex, 
    StorageContext,
    Settings,
)

# --------------------- Node Parsers  ------------------------- #
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.ingestion import IngestionPipeline


# --------------------- Schema  ------------------------- #
from llama_index.core.schema import (
    Document,
    BaseNode,
)


# --------------------- Vector Stores  ------------------------- #
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.api import AdminAPI, AsyncClientAPI, ClientAPI
from chromadb.api.models.Collection import Collection


# -------------------------- Constants --------------------- #

LLAMA_CLOUD_API_KEY = os.environ["LLAMA_CLOUD_API_KEY"]
COHERE_API_KEY = os.environ["COHERE_API_KEY"]



def run_entire_ingestion_pipeline(
        project_path: str,
        filepath: str 
    ) -> VectorStoreIndex:
    
    print("Running LLama Parser For Parsing Documets ...")
    documents = get_documents_from_llama_parser(filepath)
    nodes = transform_documents_and_insert_into_vector_store(
        documents=documents,
        project_path=project_path,
    )

    vector_store_index = get_vector_store_index(
        project_path=project_path,
    )

    return vector_store_index



# --------------------- Use LLama Parser To Read Documents  ------------------------- #

def get_documents_from_llama_parser(
        filename: str
    ) -> List[Document]:

    files = [filename]
    for file_path in files:
        documents = LlamaParse(
            api_key=LLAMA_CLOUD_API_KEY,
            result_type="markdown",
        ).load_data(
            file_path=file_path,
        )

        # Add additional metadata that is not extracted by llama parser 
        for i, doc in enumerate(documents):
            meta_data = {
                "page_label": f"{i+1}",
                "file_name" : os.path.basename(file_path)
            }

            doc.metadata = meta_data
            doc.excluded_embed_metadata_keys = ["file_name"]
            doc.excluded_llm_metadata_keys = ["file_name"]

    return documents


# --------------------- Node Post Processor  ------------------------- #

def transform_documents_and_insert_into_vector_store(
        documents: List[Document], 
        project_path: str,
    ) -> Sequence[BaseNode]:

    # Parse the documents using MarkdownElementNodeParser
    markdown_parser = MarkdownNodeParser()
    vector_store = get_chroma_vector_store(
        project_path=project_path,
    )

    pipeline = IngestionPipeline(
        transformations=[
            markdown_parser, 
            Settings.embed_model,
        ],
        vector_store=vector_store
    )

    nodes = pipeline.run(
        documents=documents,
    )

    return nodes


# -------------------------- Vector Stores ------------------------- #

def get_vector_store_index(
        project_path: str,
) -> VectorStoreIndex:
    vector_store = get_chroma_vector_store(
        project_path=project_path
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )


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