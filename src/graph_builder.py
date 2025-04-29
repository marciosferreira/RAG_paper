"""
The main goal is to build the graph
"""

# Standard Library Imports
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = 'MIG-8f2d5399-34bc-5639-8669-5d72dbfb6029'
import copy
import io
import json
import glob
import random
import logging
import re
import string
import warnings
from collections import Counter
from itertools import zip_longest
from typing import Dict, List, Literal

# Third-Party Library Imports
import faiss
import base64
import whisper
import numpy as np
import pandas as pd
import spacy
import json
import transformers
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from langchain.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from openai import OpenAI
from PIL import Image
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from typing_extensions import TypedDict


from src.config import *
from src.nodes import *

# Warnings & Logging Configuration
logging.basicConfig(filename="warnings.log", level=logging.WARNING)
warnings.simplefilter("default")  # Keeps warnings activated
warnings.showwarning = (
    lambda message, category, filename, lineno, file=None, line=None:
    logging.warning(f"{category.__name__}: {message}")
)

# Textual (metadata) search setup
# nlp = spacy.load("en_core_web_trf")  # Can test with "en_core_web_trf" (larger model)
# ruler = nlp.add_pipe("entity_ruler", before="ner")

# Embeddings & Similarity Configurations

# Model Selection (commented options for reference)
# model_id = "meta-llama/Llama-3.1-8B-Instruct"
# model_emb_name = "thenlper/gte-large"  
# tokenizer = AutoTokenizer.from_pretrained(model_id)


def build_graph():
    builder = StateGraph(State)

    # Adicionar todos os nodes
    builder.add_node("context_decision_node", context_decision_node)
    builder.add_node("use_RAG", use_RAG)
    builder.add_node("plans_comparison_tool", plans_comparison_tool)
    builder.add_node("validate_answer", validate_answer)
    builder.add_node("modify_query_node", modify_query_node)
    builder.add_node("verify_embeddings_similarity_on_FAISS", verify_embeddings_similarity_on_FAISS)
    builder.add_node("bring_conversation_history", bring_conversation_history)
    builder.add_node("create_system_message", create_system_message)
    builder.add_node("build_prompt_template", build_prompt_template)
    builder.add_node("llm_call", llm_call)
    builder.add_node("check_history_lenght", check_history_lenght)
    builder.add_node("make_history_short", make_history_short)

    builder.add_edge(START, "context_decision_node")
    builder.add_conditional_edges(
        "context_decision_node",
        decide_next_node_context  # Pass the function that decides next node
    )
    builder.add_edge("plans_comparison_tool", "create_system_message")
    builder.add_edge("use_RAG", "modify_query_node")
    builder.add_edge("modify_query_node", "verify_embeddings_similarity_on_FAISS")
    builder.add_edge("verify_embeddings_similarity_on_FAISS", "bring_conversation_history")
    builder.add_edge("bring_conversation_history", "create_system_message")
    builder.add_edge("create_system_message", "build_prompt_template") 
    builder.add_edge("build_prompt_template", "llm_call")
    builder.add_edge("llm_call", "validate_answer")
    builder.add_conditional_edges(
        "validate_answer",
        decide_next_node_validation  # Pass the function that decides next node
    )
    builder.add_conditional_edges(
        "check_history_lenght",
        decide_next_node_lenght  # Pass the function that decides next node
    )
    builder.add_edge("make_history_short", END)
    graph = builder.compile()

    # create an image of the graph and save it for debbug
    #image_bytes = graph.get_graph().draw_mermaid_png()    
    #image_path = "output_mermaid.png"
    #with open(image_path, "wb") as f:
        #f.write(image_bytes)
            
    return graph

def call_graph(graph, query, additional_metadata, thread="1"):
    # Verigy if global variable `initial_state` is set
    if 'state' not in globals():
        global initial_state 
        #creates the initial state of the graph
        initial_state = {
            "graph_state": "START",  
            "query": query,
            "embeddings": [],  
            "embeddings_loaded": False,
            "thread": thread,
            "context_chunck_history": [],
            "thread_states": {},
            "maximum_dialogs": 4,
            "maximum_context": 12,
            "final_decision": "undef",
            "context_decision": "undef",
            "adit_metadata": additional_metadata,
            #"embeddings_file_path": embeddings_file_path,
            #"full_docs_path": full_docs_path,
            "doc_texts": [],
            "use_cossine_similarity": use_cossine_similarity,
            "expanded_query": "undef",
            "extracted_metadata": [],
            "modify_query": True,
            "modified_query": "undef",
            "top_k": 5,
            "filter_by_company": False,
            "filter_by_date": False,
            "use_parent_retriever": True,
            "pdf_indice": pdf_indice
            
        }
    
    else:
        initial_state = state
        initial_state["query"] = query
        
    # Execute the graph
    result = graph.invoke(initial_state)    
    
    return result

def get_audio():

    audio_filenames = glob.glob('audios/*')
    audio_filename = random.choice(audio_filenames)
    
    audio_data = whisper.load_audio(audio_filename)
    buffer = io.BytesIO()
    np.save(buffer, audio_data)
    encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
    audio_json = {'data': encoded}
    
    return audio_json

def transcribe(audio_json):

    model = whisper.load_model("turbo")
    
    decoded_bytes = base64.b64decode(audio_json['data'])
    buffer = io.BytesIO(decoded_bytes)
    audio_array = np.load(buffer, allow_pickle=True)

    pad = whisper.pad_or_trim(audio_array)
    mel = whisper.log_mel_spectrogram(pad, n_mels=model.dims.n_mels).to(model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    del model

    return result.text

def main(mode):
    graph = build_graph()    

    # Metadados adicionais (fixos para todas as execuções)
    additional_metadata = { 
        "llm_model_name": "model_name",
        "embeddings_model_name": "model_emb_name",
        "domain": 'finance',
        "language": 'en',
    }

    print("\n Chat LLM - Type your question about your document (or 'exit' to close)\n")

    while True:
        # ask input from user
        if(mode == 'query'):
            user_input = input("\n Your question: ")
        elif(mode == 'audio'):
            audio_json = get_audio()
            user_input = transcribe(audio_json)
            print(f"Pergunta: {user_input}")

        # exit to the loop
        if user_input.lower() in ["exit", "sair", "quit"]:
            print("\n Closing the chat... See you!\n")
            break

        # call the processing graph
        state = call_graph(graph, user_input, additional_metadata, thread="1")

        # show llm response
        response = state["llm_response"]["content"]
        print("\nResponse:\n", response, "\n")
    
if __name__ == "__main__":
    main()
    



#load embeddings model defined above
#if model_emb_name not in {"text-embedding-3-small", "text-embedding-3-large"}: # do not donwload if using openai API
   # embeddings_model = SentenceTransformer(model_emb_name)