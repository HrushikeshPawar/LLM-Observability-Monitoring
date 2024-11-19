from pathlib import Path
from pydantic import BaseModel
from typing import List, Optional

class RAGConfig(BaseModel):
    query: str
    input_fname: str
    llm_name: str = "gemini-1.5-flash-002"
    temperature: float = 0
    similarity_top_k: int = 12
    node_parser_chunk_sizes: List[int] = [4096, 2048, 1024, 512]
    node_parser_chunk_overlap: int = 128
    reranker_model_name: Optional[str] = "BAAI/bge-reranker-base"
    reranker_top_n: int = 6
    # embed_model: str = "local:BAAI/bge-small-en-v1.5"
    embed_model_name: str = "models/text-embedding-004"
    data_dir: Path = Path(__file__).parent / "data"
    chat_history: List[dict] = []
