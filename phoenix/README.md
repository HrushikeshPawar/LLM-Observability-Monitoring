# Phoenix Notebooks Guide

## Overview
This directory contains Jupyter notebooks demonstrating LLM applications with Phoenix for observability and tracing, focusing on RAG implementations using LangChain and LlamaIndex.

## Prerequisites

1. Set up API keys in `.env` file at the root directory:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

2. [Optional] Start Phoenix server:
   ```bash
   phoenix serve
   ```
   Access the Phoenix UI at `http://localhost:6006`. This starts a global Phoenix server that can be used for all notebooks. Otherwise each notebook will start its own Phoenix server (which closes with the notebook).

## Notebooks

### 1. langchain-hello-world.ipynb
- Basic introduction to LangChain with Phoenix integration
- Demonstrates simple LLM interactions
- Shows Phoenix tracing and observability setup

### 2. langchain-simple-rag.ipynb
- Implements basic RAG using LangChain
- Uses Google's text embeddings
- Shows Phoenix tracing for RAG pipelines

### 3. llamaindex-hello-world.ipynb
- Introduction to LlamaIndex with Phoenix
- Demonstrates document loading and indexing
- Shows basic query operations with tracing

### 4. llamaindex-advance-rag.ipynb
- Advanced RAG implementation using LlamaIndex
- Features Sentence Window Retrieval
- Includes custom query engines and reranking
- Shows comprehensive tracing for complex workflows

## Data
The notebooks use data files from the data directory:
- paul_graham_essay.txt
- docs.txt

## Running the Notebooks

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. [Optional] Ensure Phoenix server is running on port 6006

3. Open and run notebooks in sequential order

4. Monitor traces at `http://localhost:6006`

## Important Notes

- Always verify API keys are properly set in `.env` before running notebooks
- Keep Phoenix server running throughout your experiments
- Each notebook creates its own Phoenix traces
- Traces can be viewed and analyzed in the Phoenix UI

## Dependencies
All required packages are listed in requirements.txt