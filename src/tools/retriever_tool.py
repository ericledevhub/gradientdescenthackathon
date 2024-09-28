import os

# --------------------- Core  ------------------------- #

from llama_index.core import (
    VectorStoreIndex,
)

from llama_index.core.tools import QueryEngineTool, ToolMetadata

# --------------------- Node Parsers  ------------------------- #

from llama_index.postprocessor.cohere_rerank import CohereRerank


# --------------------- Own Functions  ------------------------- #

from src.vector_stores.chroma import (
    get_chroma_vector_store,
)

COHERE_API_KEY = os.environ["COHERE_API_KEY"]


def query_engine_tool(
        project_path: str,
) -> QueryEngineTool:

    # Initialize vector store and index
    vector_store = get_chroma_vector_store(
        project_path=project_path,
    )
    vector_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
    )

    # Reank Node Processor
    cohere_rerank = CohereRerank(
        api_key=COHERE_API_KEY, 
        top_n=3
    )

    # Query Engine
    query_engine = vector_index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[cohere_rerank],
    )

    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="query_engine",
            description=(
                "Provides information about the current document."
                "Use a detailed plain text question as input to the tool."
            ),
        ),
    )

    return query_engine_tool
