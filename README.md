# Project README

## Introduction

This project explores LLM Observation and Monitoring tools. It includes various Jupyter notebooks that demonstrate how to implement llm app tracing, observaility and evalution.

## Project Structure

- `.env`: Environment variables file for storing API keys.
- `data`: Contains datasets and resources used in the project.
- `mlflow`: Contains Jupyter notebooks and scripts related to MLflow experiments.
- `requirements.txt`: Lists the Python packages required for the project.

## Setup Instructions

1. **Install Dependencies**

   Install the required Python packages by running:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up API Keys**

   Create a `.env` file in the root directory and set up your API keys as environment variables:

   ```env
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Start repective tool server**

   Before running any notebook, follow the instructions given in respective readme files to start the server.

4. **Run the Notebooks**

   Open the notebooks in the respective directory and run them as needed.

   While running the notebooks, goto `localhost:<port>` or `127.0.0.1:<port>>` on your browser to look at the traces.

## Notebooks Overview

- `langchain-hello-world.ipynb`: Introduction to using LangChain with Tracing tool for simple language model interactions.
- `langchain-simple-rag.ipynb:` Demonstrates Retrieval-Augmented Generation using LangChain and Tracing tool.
- `llamaindex-hello-world.ipynb:` Introduction to using LlamaIndex with Tracing tool for indexing and querying documents.
- `llamaindex-advance-rag.ipynb`: Advanced RAG techniques (Sentence Window Retrieval + ReRanking) using LlamaIndex, including custom query engines.


## Notes

- Ensure that you have set up the required API keys in the `.env` file before running the notebooks.
- The Tracing tool server must be running on respective ports to enable experiment tracking and tracing.
