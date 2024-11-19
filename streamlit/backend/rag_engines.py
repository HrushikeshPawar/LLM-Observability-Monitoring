import os
import yaml
import mlflow

from typing import List
from pathlib import Path

from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

from llama_index.core import Settings
from llama_index.core.llms.custom import CustomLLM
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.chat_engine import CondensePlusContextChatEngine

from utils import set_embed_model
from index_manager import get_auto_merging_retriever

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


def get_reranker(model_name:str, top_n:int) -> SentenceTransformerRerank:
    return SentenceTransformerRerank(model=model_name, top_n=top_n)

def get_retrieval_query_engine(
        data_dir:Path,
        input_fname:str,
        embed_model_name:str,
        similarity_top_k:int,
        chunk_sizes:List[int],
        chunk_overlap:int,
        reranker_model_name:str,
        reranker_top_n:int,
        llm:CustomLLM,
        ) -> RetrieverQueryEngine:
    
    # Define the retriever
    retrieval = get_auto_merging_retriever(
        data_dir=data_dir,
        input_fname=input_fname,
        embed_model_name=embed_model_name,
        chunk_overlap=chunk_overlap,
        chunk_sizes=chunk_sizes,
        similarity_top_k=similarity_top_k
    )
    print("Retriever ready! Setting node_parser and embed_model...")

    Settings.node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_sizes,
        chunk_overlap=chunk_overlap,
    )
    Settings.embed_model = set_embed_model(embed_model_name)
    Settings.llm = llm

    # Define the reranker
    node_postprocessors = []
    if reranker_model_name is not None:
        reranker = get_reranker(model_name=reranker_model_name, top_n=reranker_top_n)
        node_postprocessors.append(reranker)
        print("Reranker ready! Setting up query engine...")

    # Define the query engine
    query_engine = RetrieverQueryEngine.from_args(retriever=retrieval, node_postprocessors=node_postprocessors)
    print("Query engine ready! Returning query engine...")

    return query_engine

def get_retrieval_chat_engine(
        data_dir:Path,
        input_fname:str,
        embed_model_name:str,
        similarity_top_k:int,
        chunk_sizes:List[int],
        chunk_overlap:int,
        reranker_model_name:str,
        reranker_top_n:int,
        llm:CustomLLM,
        ) -> CondensePlusContextChatEngine:
    
    # Define the retriever
    retrieval = get_auto_merging_retriever(
        data_dir=data_dir,
        input_fname=input_fname,
        embed_model_name=embed_model_name,
        chunk_overlap=chunk_overlap,
        chunk_sizes=chunk_sizes,
        similarity_top_k=similarity_top_k
    )
    print("Retriever ready! Setting node_parser and embed_model...")

    Settings.node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_sizes,
        chunk_overlap=chunk_overlap,
    )
    Settings.embed_model = set_embed_model(embed_model_name)
    Settings.llm = llm

    # Define the reranker
    node_postprocessors = []
    if reranker_model_name is not None:
        reranker = get_reranker(model_name=reranker_model_name, top_n=reranker_top_n)
        node_postprocessors.append(reranker)
        print("Reranker ready! Setting up query engine...")

    # Define the query engine
    chat_engine = CondensePlusContextChatEngine.from_defaults(retriever=retrieval, node_postprocessors=node_postprocessors)
    print("Chat engine ready! Returning query engine...")

    return chat_engine