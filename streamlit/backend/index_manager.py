from typing import List
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, Document, StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.embeddings.gemini import GeminiEmbedding

import os
import yaml
import mlflow
import phoenix as px
from phoenix.otel import register
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from openinference.instrumentation.langchain import LangChainInstrumentor

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


# Auto-Merging Retrieval
def get_auto_merging_retriever(
        data_dir:Path,
        input_fname:str,
        embed_model_name:str,
        similarity_top_k:int,
        chunk_sizes:List[int],
        chunk_overlap:int,
    ) -> AutoMergingRetriever:

    automerge_index_path = Path(data_dir, "merging_index", f"{input_fname}-{embed_model_name.replace(':', '-').replace('/', '-')}-{similarity_top_k}-{chunk_sizes}-{chunk_overlap}")
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=chunk_sizes,
        chunk_overlap=chunk_overlap,
    )

    Settings.node_parser = node_parser
    Settings.embed_model = set_embed_model(embed_model_name)

    if not automerge_index_path.exists():
        print("Building merging index...")
        documents = SimpleDirectoryReader(input_files=[Path(data_dir, input_fname)]).load_data(show_progress=True)
        document = Document(text="\n\n".join([doc.text for doc in documents]))

        nodes = node_parser.get_nodes_from_documents([document])
        leaf_nodes = get_leaf_nodes(nodes)
        storage_context = StorageContext.from_defaults()
        storage_context.docstore.add_documents(nodes)

        automerging_index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, show_progress=True)
        automerging_index.storage_context.persist(persist_dir=automerge_index_path)
    else:
        print("Loading merging index...")
        automerging_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=automerge_index_path))
       
    automerging_retriever = automerging_index.as_retriever(similarity_top_k=similarity_top_k)

    retriever = AutoMergingRetriever(
        automerging_retriever, 
        automerging_index.storage_context, 
        verbose=True
    )

    print("Retriever ready! Returning retriever and node_parser...")

    return retriever


def set_embed_model(embed_model:str) -> str | GeminiEmbedding:
    if 'local:' in embed_model:
        print(f"Loading local embedding model: {embed_model}")
        return embed_model
    else:
        print(f"Loading Gemini embedding model: {embed_model}")
        return GeminiEmbedding(model_name=embed_model)

