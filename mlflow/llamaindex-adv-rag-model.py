import mlflow

from pathlib import Path
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor, SentenceTransformerRerank
from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings('ignore')

# Defining the CONSTANTS
DATA_DIR = Path('..', 'data')
INPUT_FPATH = Path(DATA_DIR, "eBook-How-to-Build-a-Career-in-AI.pdf")
SENTENCE_INDEX_PATH = Path(DATA_DIR, "sentence_index")

LLM_MODEL_NAME = "models/gpt4o-mini"
TEMPERATURE = 0.1

EMBED_MODEL = "local:BAAI/bge-small-en-v1.5" # "local:BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

SENTENCE_WINDOW_SIZE = 3
SIMILARITY_TOP_K = 6
RERANK_TOP_N = 2

# Setting up LLM
llm = OpenAI(model_name=LLM_MODEL_NAME, temperature=TEMPERATURE)

# Index Creation
## Load data
documents = SimpleDirectoryReader(input_files=[INPUT_FPATH]).load_data()

## Convert into a Document
document = Document(text="\n\n".join([doc.text for doc in documents]))


# Creating Sentence Index
## Create the sentence window node parser w/ default settings
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=SENTENCE_WINDOW_SIZE,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

## Llama-Index Global Settings
Settings.llm=llm
Settings.node_parser = node_parser

# Create the sentence index
# if an index file exist, then it will load it
# if not, it will rebuild it
if not SENTENCE_INDEX_PATH.exists():
    sentence_index = VectorStoreIndex.from_documents([document], embed_model=EMBED_MODEL)
    sentence_index.storage_context.persist(persist_dir=SENTENCE_INDEX_PATH)

else:
    sentence_index = load_index_from_storage(StorageContext.from_defaults(persist_dir=SENTENCE_INDEX_PATH), embed_model=EMBED_MODEL)

# Post Processing
## Instead of passing only the retrieved sentence, we pass a window of sentences - Sentence Window Retrieval
postproc = MetadataReplacementPostProcessor(target_metadata_key="window")

## Rerank the sentences using a Sentence Transformer
rerank = SentenceTransformerRerank(top_n=RERANK_TOP_N, model=RERANK_MODEL)

# Create the query engine
sentence_window_engine = sentence_index.as_query_engine(similarity_top_k=SIMILARITY_TOP_K, node_postprocessors=[postproc, rerank])

# Setting up MLFlow
mlflow.models.set_model(sentence_window_engine)