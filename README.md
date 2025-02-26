# LawGPT: LLM-based Legal ChatBot

![Python 3.12](https://img.shields.io/badge/Python-3.10-brightgreen.svg) [![ChatGPT](https://img.shields.io/badge/ChatGPT-74aa9c?logo=openai&logoColor=white)](#)  

LawGPT is a Large Language Model (LLM) based chatbot designed to provide legal information. The chatbot utilizes RAG architecture, advanced language models and embeddings to retrieve and generate contextually relevant answers from a provided legal document corpus. This project specifically focuses on the Indian Penal Code and other related legal documents.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Deployed Website](#deployed-website)


## Introduction

LawGPT aims to assist users by providing accurate and concise legal information based on the Indian Penal Code and related legal documents. The chatbot retrieves relevant context from the knowledge base to answer user queries efficiently.

## Features

- Conversational interface for querying legal information
- Uses FAISS for efficient vector search
- Embeds documents using Google Generative AI Embeddings
- Handles large document sets by splitting and batching
- Provides sources for retrieved information

## Architecture

The architecture of LawGPT includes the following components:

1. **Document Loader**: Loads legal documents from a directory of PDF files.
2. **Text Splitter**: Splits documents into manageable chunks for embedding.
3. **Embeddings**: Uses Google Generative AI Embeddings to transform text into vector representations.
4. **Vector Store**: Utilizes FAISS to store and retrieve document embeddings.
5. **LLM**: Uses the ChatGroq API to generate responses based on retrieved documents and user queries.
6. **Memory**: Maintains a conversation buffer to provide context in conversations.

### Prerequisites

- Python 3.12
- [Streamlit](https://streamlit.io/)
- [LangChain Community](https://github.com/langchain-ai/langchain-community)
- [Google Generative AI](https://github.com/google-research/google-research/tree/master/large-scale-causal-ml)
- [FAISS](https://github.com/facebookresearch/faiss)




