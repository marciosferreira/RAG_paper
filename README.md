# Introduction

This project implements a Retrieval-Augmented Generation (RAG) architecture. The system is designed to leverage a Small Language Model (SLM) locally to answer questions across any knowledge domain efficiently.

# Getting Started

Currently, the script is running smootly in NVIDIA 3060-24G,  3090-24G, NVIDIA 4090-24G.

This is a development version, you will be abble to talk about Tesla model 3 manual.

### Environment Installation

You have two option:

1) install with pip

```
pip install -r requirements.txt

```

2) Docker image

A `Dockerfile` is also provided to build the image with the requirements. Just build and run it with docker or use the provided `docker-compose.yaml` file.

```
docker compose up
```

As output a jupyterhub will be made available at `http://localhost:8888`.

### Download Weights

When running the script for the first time, the models will be downloaded and cached locally in model folder. Subsequent runs will load the model significantly faster.


#### Obtaining Meta's authorization to download the Model

1. **Obtain Meta's Authorization**:

   - Access to the model requires Meta's authorization. Visit the model's page on Hugging Face to request access: [Meta Llama 3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct). The approval process is typically quick.
   - You will need to review and accept Meta's terms before proceeding. This step is required for the model download to work.

2. **Generate a Hugging Face Access Token**

    - Go to Hugging Face Tokens Page - [link](https://huggingface.co/docs/hub/security-tokens).
    - Click New Token, set permissions to "Read", and copy the token.

3. **Login to Hugging Face**:
   - Use the following command to log in to Hugging Face before downloading the model:
     ```bash
     huggingface-cli login
     

## run the pdf ingestion if it is the dirs time using the script or the sepcific pdf:

The module data_ingestion structure

```
data_ingestion/
│── main.py
│── config.json
│── processor/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── faiss_indexer.py
```

1) The ingestion script must be run before the RAG script, as this is suposed to create the FAISS index.
   You just need to run it once, except if you want to change the pdf.

2) Insert all correct paths in: config.json and in `config.yaml`
3) start the models API by running `init.sh` script
4) Run the ingestion pipeline: python main.py --query

## Running the RAG     
To talk with the LLM about your pdf, after ingesting it, just type `python main.py --query`, a chat in the terminal will be open

> Warning: check the file config.yaml


## Important Conditions for Commercial Use of Llama models:

Redistribution and Use:
When distributing or making the Llama materials (or any derivative works) available, or a product or service that utilizes them, it is mandatory to include a copy of the license agreement. Additionally, the phrase "Developed with Meta Llama 3" must be prominently displayed on related web pages, user interfaces, blog posts, information pages, or product documentation.

Legal Compliance:
The use of Llama materials must comply with all applicable laws and regulations, including those related to commercial compliance. It is strictly prohibited to use Llama materials or any of their outputs to improve language models other than Meta Llama 3 or its derivatives.

Scalability Limitations:
If the products or services offered by the licensee, or its affiliates, have more than 700 million monthly active users in the month prior to the release of the Meta Llama 3 version, a separate license from Meta must be requested. Meta may grant this license at its sole discretion.

https://llamaimodel.com/commercial-use/
