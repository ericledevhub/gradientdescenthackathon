from typing import (
    Tuple,
)

import os, re
import chainlit as cl

# --------------------- OpenAI  ------------------------- #

from llama_index.llms.openai import OpenAI

# --------------------- LLM Chat  ------------------------- #

from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
)

# --------------------- Core  ------------------------- #

from llama_index.core import (
    Settings,
)

# --------------------- Tools ------------------------- #

from llama_index.core.tools.function_tool import FunctionTool


# --------------------- Plotly  ------------------------- #

import plotly
import plotly.express as px
import plotly.graph_objects as go


# -------------------------- Constants --------------------- #

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

GENERATION_MODEL = "gpt-3.5-turbo"


SYSTEM_INSTRUCTION_GENERATING = """You are a coding expert specializing in manipulating pandas DataFrames and generating Plotly visualizations. You will be provided with data in table form and specific user requests for plots.

Your tasks:

- Efficiently process the data using pandas.
- Create the requested Plotly plot based on user instructions.
- Display every plot with fig.show().

Generate the code in a concise python block without explanations.
"""

SYSTEM_INSTRUCTION_SELF_HEALING = """You are a coding expert in python for debugging in plotly and pandas. Correct the faulty code given the errors. Correct the code in python block."""

PLOTLY_FIGURE_OBJECT_KEY = "plotly_figure_object"
MAX_SELF_HEALING_ATTEMPTS = 3


# ------------------------------ Tool Functions -------------------------------- #

def generate_plotly_figure_tool():
    return  FunctionTool.from_defaults(
        fn=generate_plotly_figure,
    )

def agenerate_plotly_figure_tool():
    return  FunctionTool.from_defaults(
        async_fn=agenerate_plotly_figure,
    )


def generate_plotly_figure(
        user_request: str, 
        table: str,
    ) -> str:
    """
    Generates a Plotly figure based on a user's request and a provided table in text format.

    This function takes a user's request and a table (in text format), generates the 
    corresponding Plotly code, and attempts to execute it. If successful, the Plotly 
    figure is stored in the session state; otherwise, an error message detailing the 
    failure is returned.

    Parameters:
    ----------
    user_request : str
        A string containing the user request, including data and specifications 
        for the desired Plotly plot.

    table : str
        A string containing the table data in a text format, which will be used 
        to generate the Plotly figure.

    Returns:
    -------
    str
        A message indicating success or detailing any errors encountered during 
        figure generation.
    """
    
    # Generate Plotly code from the user request
    chat_response = generate_plotly_code_llm_call(
        user_query=user_request,
        table=table,
    )
    code_message = chat_response.message.content
    
    # Execute the generated code and capture the figure and any errors
    fig, error = execute_plotly_code(code_message)
    code = extract_python_code(code_message)


    if error is not None:
        # If an error occurred, clear the session state for the figure
        cl.user_session.set(
            key=PLOTLY_FIGURE_OBJECT_KEY, 
            value=None,
        )

        return (
            "Failed to generate Plotly figure.\n\n"
            "Tried executing the following code:\n\n"
            f"```python\n{code}\n```\n\n"
            f"Following error encountered:\n\n{error}"
        )

    # If execution was successful, store the figure in the user session state
    cl.user_session.set(
        key=PLOTLY_FIGURE_OBJECT_KEY, 
        value=fig,
    )

    return (
        "Successfully generated a Plotly figure.\n\n"
        "Executed following code:\n\n"
        f"```python\n{code}\n```\n\n"
    )


async def agenerate_plotly_figure(
    user_request: str, 
    table: str,
    ) -> str:    
    """
    Generates a Plotly figure based on a user's request and a provided table in text format.

    This function takes a user's request and a table (in text format), generates the 
    corresponding Plotly code, and attempts to execute it. If successful, the Plotly 
    figure is stored in the session state; otherwise, an error message detailing the 
    failure is returned.

    Parameters:
    ----------
    user_request : str
        A string containing the user request, including data and specifications 
        for the desired Plotly plot.

    table : str
        A string containing the table data in a text format, which will be used 
        to generate the Plotly figure.

    Returns:
    -------
    str
        A message indicating success or detailing any errors encountered during 
        figure generation.
    """
    
    # Generate Plotly code from the user request
    chat_response = await agenerate_plotly_code_llm_call(
        user_query=user_request,
        table=table,
    )
    code_message = chat_response.message.content
    
    # Execute the generated code and capture the figure and any errors
    fig, error = await aexecute_plotly_code(code_message)
    code = extract_python_code(code_message)


    if error is not None:
        # If an error occurred, clear the session state for the figure
        cl.user_session.set(
            key=PLOTLY_FIGURE_OBJECT_KEY, 
            value=None)

        return (
            "Failed to generate Plotly figure.\n\n"
            "Tried executing the following code:\n\n"
            f"```python\n{code}\n```\n\n"
            f"Following error encountered:\n\n{error}"
        )

    # If execution was successful, store the figure in the user session state
    cl.user_session.set(
        key=PLOTLY_FIGURE_OBJECT_KEY, 
        value=fig,
    )

    return (
        "Successfully generated a Plotly figure.\n\n"
        "Executed following code:\n\n"
        f"```python\n{code}\n```\n\n"
    )


# ------------------------------ Sub Functions of Tool functions -------------------------------- #
def generate_plotly_code_llm_call(
        user_query: str,
        table: str
    ) -> Tuple:
    # Prepare the messages for the LLM with the initial user request
    messages = [
        ChatMessage(role="system", content=SYSTEM_INSTRUCTION_GENERATING),
        ChatMessage(role="user", content=f"User Request: {user_query}\n\n Table: {table}"),
    ]

    llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model=GENERATION_MODEL, 
        temperature=0.1,
    )

    # First attempt to get code from LLM
    chat_response = llm.chat(
        messages=messages, 
    )
    code_message = chat_response.message.content
    
    # Execute the generated Plotly code and extract Python code for possible further use
    fig, error = execute_plotly_code(code_message)
    code = extract_python_code(code_message)
    
    # Self-healing loop in case of errors
    if error is not None:
        error_loop = True
        i = 0
        
        while i < MAX_SELF_HEALING_ATTEMPTS and error_loop:
            print(f"Self-healing attempt: {i + 1}")
            
            # Prepare a message with the faulty code and the encountered error for LLM
            content = f"Faulty Code: ```python\n{code}\n``` \n\n Following error encountered: {error}"
            messages = [
                ChatMessage(role="system", content=SYSTEM_INSTRUCTION_SELF_HEALING),
                ChatMessage(role="user", content=content)
            ]
            
            # Ask LLM to correct the code
            chat_response = llm.chat(messages=messages)
            code_message = chat_response.message.content
            code = extract_python_code(code_message)
            
            # Try executing the new code
            _, error = execute_plotly_code(code_message)
            
            if error is None:
                error_loop = False  # Exit the loop if no error
                print("Self-healing successful!")
            else:
                i += 1  # Continue to the next iteration if there's still an error

        if error_loop:
            print("Self-healing failed after maximum attempts.")

    return chat_response



async def agenerate_plotly_code_llm_call(
        user_query: str,
        table: str
    ) -> ChatResponse:
    messages = [
        ChatMessage(role="system", content=SYSTEM_INSTRUCTION_GENERATING),
        ChatMessage(role="user", content=f"User Request: {user_query}\n\n Table: {table}"),
    ]

    llm = OpenAI(
        api_key=OPENAI_API_KEY,
        model=GENERATION_MODEL, 
        temperature=0.1,
    )

    chat_response = await llm.achat(
        messages=messages, 
    )
    code_message = chat_response.message.content

    # Execute the generated Plotly code and extract Python code for possible further use
    fig, error = await aexecute_plotly_code(code_message)
    code = extract_python_code(code_message)

    # Self-healing loop in case of errors
    if error is not None:
        error_loop = True
        i = 0
        
        while i < MAX_SELF_HEALING_ATTEMPTS and error_loop:
            print(f"Self-healing attempt: {i + 1}")
            
            # Prepare a message with the faulty code and the encountered error for LLM
            content = f"Faulty Code: ```python\n{code}\n``` \n\n Following error encountered: {error}"
            messages = [
                ChatMessage(role="system", content=SYSTEM_INSTRUCTION_SELF_HEALING),
                ChatMessage(role="user", content=content)
            ]
            
            # Ask LLM to correct the code
            chat_response = await llm.achat(messages=messages)
            code_message = chat_response.message.content
            code = extract_python_code(code_message)
            
            # Try executing the new code
            _, error = await aexecute_plotly_code(code_message)
            
            if error is None:
                error_loop = False  # Exit the loop if no error
                print("Self-healing successful!")
            else:
                i += 1  # Continue to the next iteration if there's still an error

        if error_loop:
            print("Self-healing failed after maximum attempts.")

    return chat_response


def execute_plotly_code(python_code: str) -> Tuple[plotly.graph_objs.Figure, None]:
    # Remove backticks and fig.show()
    plotly_code = sanitize_plotly_code(extract_python_code(python_code))
    
    # Try to execute the code
    ldict = {"px": px, "go": go}
    error_message = None  # Initialize the error_message as None

    try:
        exec(plotly_code, globals(), ldict)  # Execute the Plotly code in the local context
        fig = ldict.get("fig", None)  # Retrieve the 'fig' object if it's created
    except Exception as e:
        error_message = str(e)  # Capture the exception message
        fig = None  # Ensure 'fig' is None in case of an error

    if fig is not None:
        fig.update_layout(
            template="plotly_dark"
        )

    # Return both the figure (or None) and the error message (or None)
    return fig, error_message


async def aexecute_plotly_code(python_code: str) -> Tuple[plotly.graph_objs.Figure, None]:
    # Remove backticks and fig.show()
    plotly_code = sanitize_plotly_code(extract_python_code(python_code))
    
    # Try to execute the code
    ldict = {"px": px, "go": go}
    error_message = None  # Initialize the error_message as None

    try:
        exec(plotly_code, globals(), ldict)  # Execute the Plotly code in the local context
        fig = ldict.get("fig", None)  # Retrieve the 'fig' object if it's created
    except Exception as e:
        error_message = str(e)  # Capture the exception message
        fig = None  # Ensure 'fig' is None in case of an error

    if fig is not None:
        fig.update_layout(
            template="plotly_dark"
        )

    # Return both the figure (or None) and the error message (or None)
    return fig, error_message


# ------------------------- Utils/help functions ------------------------------ #

def sanitize_plotly_code(raw_plotly_code: str) -> str:
    # Remove the fig.show() statement from the plotly code
    plotly_code = raw_plotly_code.replace("fig.show()", "")

    return plotly_code


def extract_python_code(markdown_string: str) -> str:
    # Regex pattern to match Python code blocks
    pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

    # Find all matches in the markdown string
    matches = re.findall(pattern, markdown_string, re.IGNORECASE)

    # Extract the Python code from the matches
    python_code = []
    for match in matches:
        python = match[0] if match[0] else match[1]
        python_code.append(python.strip())

    if len(python_code) == 0:
        return markdown_string

    return python_code[0]