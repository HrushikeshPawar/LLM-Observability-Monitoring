import os
import yaml
import logging
import requests

import streamlit as st

from pathlib import Path
from typing import List, Dict, Optional

# Setup Logging
logging.basicConfig(level=logging.DEBUG, filename="eval.log", encoding="utf-8")
logger = logging.getLogger(__name__)

# Load the configuration file
config_path = os.environ.get("CONFIG_PATH", Path(__file__).parent.parent / "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# FastAPI URL
fastapi_url = config["fastapi_url"]

# FastAPI endpoints - Query Chat Engine
def query_fastapi_chat_engine(
        query: str,
        input_fname: str,
        llm_name: str,
        llm_temperature: float,
        embed_model_name: str,
        similarity_top_k: int,
        node_parser_chunk_sizes: list,
        node_parser_chunk_overlap: list,
        reranker_model_name: str,
        reranker_top_n: int,
        data_dir: str,
        chat_history: List[Dict[str, Optional[str]]] = [],
    ):
    
    # Setup endpoint
    url = f"{fastapi_url}/query_chat_engine"

    # Setup payload
    payload = {
        "query": query,
        "input_fname": input_fname,
        "llm_name": llm_name,
        "temperature": llm_temperature,
        "embed_model_name": embed_model_name,
        "similarity_top_k": similarity_top_k,
        "node_parser_chunk_sizes": node_parser_chunk_sizes,
        "node_parser_chunk_overlap": node_parser_chunk_overlap,
        "reranker_model_name": reranker_model_name,
        "reranker_top_n": reranker_top_n,
        "data_dir": data_dir,
        "chat_history": chat_history,
    }

    # Headers
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
    }

    # Add Debug Statements
    logger.debug(f"URL: {url}")
    logger.debug(f"Payload: {payload}")
    logger.debug(f"Headers: {headers}")

    # Make the request
    response = requests.post(url, headers=headers, json=payload, timeout=180)
    logger.debug(f"Response Status COde: {response.status_code}")

    if response.status_code != 405:
        response.raise_for_status()
        return response.json()
    else:
        logger.error(f"Response Content: {response.text}")
        st.write(f"Response Content: {response.text}")
        st.error("Method Not Allowed. Please check the FastAPI server logs for more information.")
    
    return None


# Streamlit App
st.set_page_config(page_title="LLM Observability and Monitoring", page_icon="🤖", layout="wide")

# Sidebar
st.sidebar.header(":hammer_and_wrench: RAG Config", divider=True)

st.sidebar.subheader(":robot_face: LLM Config")
llm_name = st.sidebar.selectbox("LLM Model", config["available_llm_models"])
llm_temperature = st.sidebar.slider("LLM Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
st.sidebar.divider()

st.sidebar.subheader(":file_folder: Vector Index Config")
input_fname = st.sidebar.selectbox("Input File Name", options=config["available_input_fnames"])
embed_model_name = st.sidebar.selectbox(
    "Embed Model Name",
    options=["models/" + m for m in config["available_embed_models"]],
    format_func=lambda x: x.split("/")[-1]
    )

st.sidebar.divider()

# Main Content
st.header("LLM Observability Demo")
st.write("This is a demo for the LLM Observability, showcasing what LLM tracing looks like, using MLFlow and Arize-Pheonix.")
st.divider()

# Setup up Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Scrollable component for Chat History
chat_container = st.container()
with chat_container:
    chat_bar = st.columns([.9, .1])

    with chat_bar[0]:
        st.subheader(":speech_balloon: Chat History")
    
    with chat_bar[1]:
        clear_chat = st.button("", icon=":material/restart_alt:", help="Clear Chat History", type="primary", on_click=lambda: st.session_state.pop("chat_history", None))
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

# Chat Input
chat_input = st.chat_input("Ask your query")


if chat_input:
    with chat_container:
        with st.chat_message("user"):
            st.write(chat_input)
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Query the FastAPI Chat Engine
                    response = query_fastapi_chat_engine(
                        query=chat_input,
                        input_fname=input_fname,
                        llm_name=llm_name,
                        llm_temperature=llm_temperature,
                        embed_model_name=embed_model_name,
                        similarity_top_k=config["similarity_top_k"],
                        node_parser_chunk_sizes=config["node_parser_chunk_sizes"],
                        node_parser_chunk_overlap=config["node_parser_chunk_overlap"],
                        reranker_model_name=config["reranker_model_name"],
                        reranker_top_n=config["reranker_top_n"],
                        data_dir=config["data_dir"],
                        chat_history=st.session_state.chat_history
                    )

                    if response:
                        assistant_response = response.get("response", "I'm sorry, I don't have an answer for that.")
                        # st.write(assistant_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
                        st.rerun()
                    
                    else:
                        st.error("Failed to get a response from the FastAPI Chat Engine server. Please check the server status and try again.")

                except Exception as e:
                    logger.error(f"Error: {e}")
                    st.error(f"An error occurred: {e}")
                    response = None