
import os
import asyncio
import chainlit as cl

from src.ingestion.llama_parser_async import arun_entire_ingestion_pipeline
from src.ingestion.llama_parser import run_entire_ingestion_pipeline

# --------------------- Core  ------------------------- #
from llama_index.core import (
    Settings,
)

# --------------------- LLM Chat  ------------------------- #

from llama_index.core.callbacks import CallbackManager

# --------------------- LLM Agent  ------------------------- #

from llama_index.core.agent import ReActAgent


# --------------------- OpenAI  ------------------------- #

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

# --------------------- Own Functions  ------------------------- #

from src.tools.retriever_tool import query_engine_tool
from src.tools.plotly_tool import agenerate_plotly_figure_tool
from src.callbacks.chainlit import LlamaIndexCallbackHandler


# --------------------- Constants  ------------------------- #

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

EMBEDDING_MODEL  = "text-embedding-3-small"
GENERATION_MODEL = "gpt-3.5-turbo"

PLOTLY_FIGURE_OBJECT_KEY = "plotly_figure_object"


WELCOME_MESSAGE = """Welcome to the React Agent With RAG and Plot Tools! To get started:
1. Upload a PDF of your choice.
2. Ask any questions related to the content of the file.
"""


# --------------------- On start ask user for a file  ------------------------- #

async def ask_user_file():
    files = None
    while files is None:
        # Ask user to upload a file
        files = await cl.AskFileMessage(
            content=WELCOME_MESSAGE,
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    # Inform user about processing
    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    # Simulate spinner and processing
    spinner_task = asyncio.create_task(simulate_processing(msg, file))

    try:
        # Simulate a processing delay, replace this with actual logic
        run_entire_ingestion_pipeline(
            project_path=PROJECT_PATH,
            file_path=file.path,
        )

    finally:
        # Cancel spinner task when done processing
        spinner_task.cancel()
        try:
            await spinner_task
        except asyncio.CancelledError:
            pass

    # Inform user that processing is complete
    msg.content = f"The file has been successfully processed. You can view it here {file.name} and start asking questions!"
    msg.elements = [
        cl.Pdf(
            name=file.name,
            path=file.path,
            display="side",
        )
    ]

    await msg.update()


# Function to simulate spinner during processing
async def simulate_processing(msg, file):
    spinner = ['|', '/', '-', '\\']
    i = 0
    try:
        while True:
            msg.content = f"Processing `{file.name}`... {spinner[i % len(spinner)]}"
            await msg.update()
            i += 1
            await asyncio.sleep(0.5)  # Adjust the speed of the spinner
    except asyncio.CancelledError:
        # Handle task cancellation (e.g., when the processing is done)
        pass


# --------------------- On start agent setup  ------------------------- #

async def agent_setup():

    # LLama Index Global Settings
    Settings.llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model=GENERATION_MODEL, 
    )

    Settings.embed_model = OpenAIEmbedding(
        api_key=OPENAI_API_KEY,
        model=EMBEDDING_MODEL,
    )

    # Chainlit Callback Manager
    Settings.callback_manager = CallbackManager(
        [LlamaIndexCallbackHandler()],
    )

    # Tools     
    tools = [
        query_engine_tool(
            project_path=PROJECT_PATH,
        ),
        agenerate_plotly_figure_tool(),
    ]

    # React Agent 
    react_agent = ReActAgent.from_tools(
        tools=tools,
        llm=Settings.llm,
        verbose=True,
    )

    # Store necessary objects in the user session
    cl.user_session.set("react_agent", react_agent)



# Start function to prompt user for file upload
@cl.on_chat_start
async def start():
    await ask_user_file()
    await agent_setup()


@cl.on_message
async def main(message: cl.Message):
    # Initialize an empty list for elements if needed
    cl.user_session.set(
        key=PLOTLY_FIGURE_OBJECT_KEY, 
        value=None
    )
    elements = []

    # Fetch React Agent from user_session
    react_agent: ReActAgent = cl.user_session.get("react_agent")  

    # Get Response 
    response_message = cl.Message(content="")
    response = await react_agent.achat(
        message=message.content,
    )

    # Stream Response
    for token in response.response:
        await response_message.stream_token(token=token)


    # Try to retrieve the Plotly figure from the session
    plotly_figure = cl.user_session.get(PLOTLY_FIGURE_OBJECT_KEY)

    # Only add the Plotly element if the figure exists
    if plotly_figure is not None:
        elements.append(
            cl.Plotly(
                figure=plotly_figure,
                display="inline",
            )
        )
        response_message.elements = elements

    await response_message.send()