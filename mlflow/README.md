# MLflow Notebooks Guide

## Overview
This directory contains Jupyter notebooks demonstrating LLM applications with MLflow for experiment tracking, focusing on RAG implementations using LangChain and LlamaIndex.

## Prerequisites

1. Set up API keys in `.env` file at the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

2. Start MLflow server:
   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```
   Access the MLflow UI at `http://localhost:5000`

## Notebooks

### 1. langchain-hello-world.ipynb
- Basic introduction to LangChain with MLflow integration
- Demonstrates simple LLM interactions
- Shows MLflow experiment tracking setup

### 2. langchain-simple-rag.ipynb
- Implements basic RAG using LangChain
- Uses Google's text embeddings
- Shows MLflow experiment logging for RAG pipelines

### 3. llamaindex-hello-world.ipynb
- Introduction to LlamaIndex with MLflow
- Demonstrates document loading and indexing
- Shows basic query operations

### 4. llamaindex-advance-rag.ipynb
- Advanced RAG implementation using LlamaIndex
- Features Sentence Window Retrieval
- Includes custom query engines and reranking

## Data
The notebooks use data files from the data directory:
- paul_graham_essay.txt
- docs.txt

## Running the Notebooks

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure MLflow server is running on port 5000

3. Open and run notebooks in sequential order

4. Monitor experiments at `http://localhost:5000`

## Important Notes

- Always verify API keys are properly set in `.env` before running notebooks
- Keep MLflow server running throughout your experiments
- Each notebook creates its own MLflow experiment for tracking
- Experiments can be viewed and compared in the MLflow UI

## Dependencies
All required packages are listed in requirements.txt