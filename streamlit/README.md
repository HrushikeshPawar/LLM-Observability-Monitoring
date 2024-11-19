# Streamlit LLM Observability Demo Guide

## Overview
This directory contains a Streamlit demo application demonstrating LLM Observability using a RAG (Retrieval Augmented Generation) application with LLM tracing using [MLflow](https://mlflow.org/docs/latest/llms/tracing/index.html) and [Phoenix](https://docs.arize.com/phoenix/tracing/llm-traces). The app includes a [FastAPI](https://fastapi.tiangolo.com/) backend and [Streamlit](https://streamlit.io/) frontend.

## Prerequisites

1. Install dependencies:
   ```bash
   # Install base requirements
   pip install -r requirements.txt
   
   # Install additional packages with --no-deps
   pip install --no-deps llama-index-llms-gemini llama-index-embeddings-gemini
   ```

2. Setup environment variables in `.env`:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   PHOENIX_HOST=your_phoenix_host
   PHOENIX_PORT=your_phoenix_port
   ```

3. Update `config.yaml`:
   - Set correct `data_dir` path
   - Configure LLM models and embedding models
   - Adjust chunking parameters if needed
   - Verify ports for FastAPI and Streamlit

## Project Structure

- `backend/`: FastAPI server implementation
  - `main.py`: Main FastAPI application
  - `utils.py`: Utility functions
  - `models.py`: Pydantic models
  - `rag_engines.py`: RAG implementation
  - `rest-api-test.http`: API test file
- `ui/`: Streamlit frontend
  - `main.py`: Streamlit interface
- `config.yaml`: Configuration settings

## Running the Application

1. Start MLflow server:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. Start Phoenix server:
   ```bash
   phoenix serve
   ```

3. Start FastAPI backend (from backend directory):
   ```bash
   cd backend
   fastapi dev main.py
   ```
   Test APIs using `rest-api-test.http`

4. Start Streamlit UI (from ui directory):
   ```bash
   cd ui
   streamlit run main.py
   ```

## Access Points

- MLflow UI: http://localhost:5000
- Phoenix UI: http://localhost:6006
- FastAPI: http://localhost:8000
- Streamlit UI: http://localhost:8501

## Features

- RAG implementation with LangChain and LlamaIndex
- Support for multiple LLM models (Gemini, GPT)
- Configurable chunking and embedding settings
- Real-time tracing with MLflow and Phoenix
- User feedback collection
- Comprehensive logging

## Important Notes

- Ensure all servers (MLflow, Phoenix, FastAPI) are running before starting Streamlit
- Check logs in `eval.log` for debugging
- Monitor traces in MLflow and Phoenix UIs
- Use the sidebar in Streamlit to configure RAG parameters
- Test API endpoints before running the UI

## Dependencies
See `requirements.txt` for complete list of dependencies.

## Configuration
See `config.yaml` for available settings:
- LLM models
- Embedding models
- Chunking parameters
- Server configurations
- Tracing settings