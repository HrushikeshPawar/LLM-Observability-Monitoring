from llama_index.llms.gemini import Gemini
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding


def set_embed_model(embed_model_name:str) -> str | GeminiEmbedding:
    if 'local:' in embed_model_name:
        print(f"Loading local embedding model: {embed_model_name}")
        return embed_model_name
    elif "Google/" in embed_model_name:
        embed_model_name = embed_model_name.replace("Google/", "")
        print(f"Loading Gemini embedding model: {embed_model_name}")
        return GeminiEmbedding(model_name=embed_model_name)
    elif "OpenAI/" in embed_model_name:
        embed_model_name = embed_model_name.replace("OpenAI/", "").replace("models/", "")
        print(f"Loading OpenAI embedding model: {embed_model_name}")
        return OpenAIEmbedding(model=embed_model_name)

    raise ValueError(f"Invalid embed_model_name: {embed_model_name}")


def set_llm(model_name: str, temperature:float) -> Gemini | OpenAI:
    if "gemini" in model_name.lower():
        return Gemini(llm_name=model_name, temperature=temperature)
    elif "gpt" in model_name.lower():
        return OpenAI(model=model_name, temperature=temperature)
