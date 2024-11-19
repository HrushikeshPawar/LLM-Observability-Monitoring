import os
import logging
import uvicorn
import yaml

from fastapi import FastAPI
from pathlib import Path

from llama_index.core import Response, Settings
from llama_index.llms.gemini import Gemini
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.base.llms.types import ChatMessage

import mlflow
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

from models import RAGConfig
from rag_engines import get_retrieval_query_engine, get_retrieval_chat_engine, set_embed_model

# Configure logging
logging.basicConfig(filename="eval.log", encoding="utf-8", level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the configuration file
config_path = os.environ.get("CONFIG_PATH", Path(__file__).parent.parent / "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Setup Tracing
mlflow.set_tracking_uri(config['mlflow_tracking_uri'])
mlflow.set_experiment(config['tracing_project_name'])
mlflow.llama_index.autolog()
mlflow.langchain.autolog()

# Setup Phoenix tracer
tracer_provider = register(project_name=config['tracing_project_name'])

# Initialize the LLamaIndex and LangChain Instrumentor
LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


# Define the FastAPI app
app = FastAPI()

# Define the RAG Query Engine route
@app.post("/rag_query_engine", tags=["RAG Query Engine"])
async def query_engine(
    rag_config: RAGConfig,
) -> dict:
    query_engine = get_retrieval_query_engine(
        data_dir=rag_config.data_dir,
        input_fname=rag_config.input_fname,
        embed_model_name=rag_config.embed_model_name,
        similarity_top_k=rag_config.similarity_top_k,
        chunk_sizes=rag_config.node_parser_chunk_sizes,
        chunk_overlap=rag_config.node_parser_chunk_overlap,
        reranker_model_name=rag_config.reranker_model_name,
        reranker_top_n=rag_config.reranker_top_n,
        llm=Gemini(llm_name=rag_config.llm_name, temperature=rag_config.temperature),
    )
    print("Query engine ready! Setting node_parser and embed_model...")

    Settings.node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=rag_config.node_parser_chunk_sizes, chunk_overlap=rag_config.node_parser_chunk_overlap)
    Settings.embed_model = set_embed_model(rag_config.embed_model_name)
    Settings.llm = Gemini(llm_name=rag_config.llm_name, temperature=rag_config.temperature)
    query_engine:RetrieverQueryEngine

    print("Query engine ready! Running query...")
    response:Response = await query_engine.aquery(rag_config.query)
    print("Query complete! Returning response...")

    return {"response": response.response, "retrieved_chunks": [r.node.text for r in response.source_nodes]}


# Define the RAG Chat Engine route
@app.post("/query_chat_engine", tags=["RAG Chat Engine"])
async def query_chat_engine(
    rag_config: RAGConfig,
) -> dict:
    chat_engine = get_retrieval_chat_engine(
        data_dir=rag_config.data_dir,
        input_fname=rag_config.input_fname,
        embed_model_name=rag_config.embed_model_name,
        similarity_top_k=rag_config.similarity_top_k,
        chunk_sizes=rag_config.node_parser_chunk_sizes,
        chunk_overlap=rag_config.node_parser_chunk_overlap,
        reranker_model_name=rag_config.reranker_model_name,
        reranker_top_n=rag_config.reranker_top_n,
        llm=Gemini(llm_name=rag_config.llm_name, temperature=rag_config.temperature),
    )
    print("Chat engine ready! Setting node_parser and embed_model...")

    Settings.node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=rag_config.node_parser_chunk_sizes, chunk_overlap=rag_config.node_parser_chunk_overlap)
    Settings.embed_model = set_embed_model(rag_config.embed_model_name)
    Settings.llm = Gemini(llm_name=rag_config.llm_name, temperature=rag_config.temperature)
    chat_engine:CondensePlusContextChatEngine

    for msg in rag_config.chat_history:
        chat_engine.chat_history.append(ChatMessage(**msg))

    print("Chat engine ready! Running query...")
    response:Response = await chat_engine.achat(rag_config.query)
    print("Chat complete! Returning response...")

    return {"response": response.response, "retrieved_chunks": [r.node.text for r in response.source_nodes], 'messages': [msg.dict() for msg in chat_engine.chat_history]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8033)