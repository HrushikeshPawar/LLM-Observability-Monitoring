import os
import yaml
import httpx
import logging
import requests

import phoenix as px
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from pathlib import Path
from time import sleep
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

# HTTPX Client
httpx_client = httpx.Client()

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

# Get Phoenix Span ID
def get_current_span_id() -> str:
    sleep(0.5)
    # Get span dataframe from the Project
    spans_df = px.Client().get_spans_dataframe(project_name=config["tracing_project_name"], root_spans_only=True)
    spans_df.sort_values(by="end_time", ascending=False, inplace=True)
    return spans_df.index.values[0]



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

st.sidebar.subheader(":grey[:material/manufacturing:] More Settings")
allow_user_feedback = st.sidebar.toggle("User Feedback", value=False)


# Main Content
st.header("LLM Observability Demo")
st.write("This is a demo for the LLM Observability, showcasing what LLM tracing looks like, using MLFlow and Arize-Pheonix.")
st.divider()

# Setup up Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.dialog("Give Feedback")
def take_feedback(icon:str, color:str, score:int, msg_idx:int):
    st.write(f"Span ID: `{st.session_state.chat_history[msg_idx]['span_id']}`")
    st.write(f"Selected Feedback: :{color}[{icon}]")
    explanation = st.text_area("Want to add something?")
    if st.columns(3)[1].button("Submit", use_container_width=True):
        st.session_state.chat_history[msg_idx]['feedback'] = {"score": score, "explanation": explanation}

        annotation_payload = {
            "data": [
                {
                    "span_id": st.session_state.chat_history[msg_idx]['span_id'],
                    "name": "user_feedback",
                    "annotator_kind": "HUMAN",
                    "result": {
                        "label": "thumbs-up" if score == 1 else "thumbs-down",
                        "score": score,
                        "explanation": st.session_state.chat_history[msg_idx]['feedback']['explanation']
                    },
                    "metadata": {},
                }
            ]
        }
        print(annotation_payload)

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
            }

        httpx_client.post(
            url=f"http://{os.environ.get('PHOENIX_HOST')}:{os.environ.get('PHOENIX_PORT')}/v1/span_annotations?sync=false",
            json=annotation_payload,
            headers=headers
        )

        st.rerun()

@st.dialog("Give Feedback")
def show_feedback(icon:str, color:str, msg_idx:int):
    st.write(f"Span ID: `{st.session_state.chat_history[msg_idx]['span_id']}`")
    st.write(f"Selected Feedback: :{color}[{icon}]")
    st.text_area("Want to add something?", value=st.session_state.chat_history[msg_idx]['feedback']['explanation'], disabled=True)
    
    button_cols = st.columns([1, 2, 1, 2, 1])
    
    if button_cols[1].button("Close", use_container_width=True):
        st.rerun()
    
    if button_cols[3].button("Reset", use_container_width=True):
        st.session_state.chat_history[msg_idx]['feedback'] = None
        st.rerun()

# Scrollable component for Chat History
chat_container = st.container()
with chat_container:
    chat_bar = st.columns([.9, .1])

    with chat_bar[0]:
        st.subheader(":speech_balloon: Chat History")
    
    with chat_bar[1]:
        clear_chat = st.button("", icon=":material/restart_alt:", help="Clear Chat History", type="primary", on_click=lambda: st.session_state.pop("chat_history", None))
    
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if allow_user_feedback:
                if msg["role"] == "assistant" or msg["role"] == "ai":
                    if msg['feedback'] is None:
                        button_cols = st.columns([.05, .05, .9])

                        with button_cols[0]:
                            with stylable_container(
                                key="green_button",
                                css_styles="""
                                    button {
                                        color: green;
                                        border: 0px;
                                    }
                                    button:focus {
                                        color: green;
                                        border: 0px;
                                    }
                                    """,
                            ):
                                if st.button("", icon=":material/thumb_up:", key=f"green_button_{idx}"):
                                    take_feedback(icon=":material/thumb_up:", color="green", score=1, msg_idx=idx)

                        with button_cols[1]:
                            with stylable_container(
                                key="red_button",
                                css_styles="""
                                    button {
                                        color: red;
                                        border: 0px;
                                    }
                                    """,
                            ):
                                if st.button("", icon=":material/thumb_down:", key=f"red_button_{idx}"):
                                    take_feedback(icon=":material/thumb_down:", color="red", score=0, msg_idx=idx)
                    else:
                        if msg['feedback']['score'] == 1:
                            with stylable_container(
                                    key="green_button",
                                    css_styles="""
                                        button {
                                            color: green;
                                            border: 0px;
                                        }
                                        button:focus {
                                            color: green;
                                            border: 0px;
                                        }
                                        """,
                                ):
                                    if st.button("", icon=":material/thumb_up:", key=f"green_button_{idx}"):
                                        show_feedback(icon=":material/thumb_up:", color="green", msg_idx=idx)
                        else:
                            with stylable_container(
                                key="red_button",
                                css_styles="""
                                    button {
                                        color: red;
                                        border: 0px;
                                    }
                                    """,
                            ):
                                if st.button("", icon=":material/thumb_down:", key=f"red_button_{idx}"):
                                    show_feedback(icon=":material/thumb_down:", color="red", msg_idx=idx)

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
                        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response, "feedback": None, "span_id": get_current_span_id()})
                        st.rerun()
                    
                    else:
                        st.error("Failed to get a response from the FastAPI Chat Engine server. Please check the server status and try again.")

                except Exception as e:
                    logger.error(f"Error: {e}")
                    st.error(f"An error occurred: {e}")
                    response = None