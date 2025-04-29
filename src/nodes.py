import os
import json
import copy
import sys
from pathlib import Path
# Third-Party Imports
import numpy as np
import torch
import transformers
from typing import List, Dict, Literal
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.vectorstores import FAISS deprecated
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
from langchain.tools import BaseTool
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.storage import InMemoryStore


# Custom Imports
try:
    from phoenix_decorator import phoenix_trace_function
    from utils import State
    from config import *
    from model_client import MLModelAPIClient
except:
    from .phoenix_decorator import phoenix_trace_function
    from .utils import State
    from .config import *
    from .model_client import MLModelAPIClient

# Se eos_token_id for None, usa o pad_token_id como fallback
# tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
# pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Ativar quantização em 4 bits
#     bnb_4bit_use_double_quant=True,  # (Opcional) Melhor uso da memória
#     bnb_4bit_quant_type="nf4",      # Tipo de quantização (default: "nf4")
#     bnb_4bit_compute_dtype=torch.float16  # Define o dtype como float16
# )

# pipeline = transformers.pipeline(
#     "text-generation", 
#     do_sample=False,
#     torch_dtype=torch.float16,
#     model=llm_model_path,
#     pad_token_id=pad_token_id, 
#     model_kwargs={
#         "device_map": "auto",
#         "quantization_config": quantization_config,   # Ativar 4 bits           
#     },
# )


ml_client = MLModelAPIClient()
    
# faiss_index_path = faiss_index_path_config 
print(faiss_index_path)
if os.path.exists(faiss_index_path):
    vectorstore = FAISS.load_local(faiss_index_path, ml_client.get_embedding, allow_dangerous_deserialization=True)    
else:
    print(" ERROR: FAISS index was not saved or is empty.")


print(parent_json_path)

# 🔹 Carregar os chunks
if os.path.exists(parent_json_path) and os.path.exists(child_json_path):
    #print("📂 Arquivos JSON encontrados. Carregando chunks...")
    with open(parent_json_path, "r", encoding="utf-8") as f:
        parent_chunks = json.load(f)
    with open(child_json_path, "r", encoding="utf-8") as f:
        child_chunks_raw = json.load(f)
    child_chunks = [Document(page_content=doc["content"], metadata=doc["metadata"]) for doc in child_chunks_raw]
else:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Please run data ingestion first")
    sys.exit(1)  # Encerra o script com código de erro (1)
   


if use_parent_retriever:
    parent_store = InMemoryStore()
    parent_store.mset([(doc["chunk_id"], doc) for doc in parent_chunks])
    #print("\n✅ Parent retriever ativado: estrutura de chunks pais e filhos criada.")
else:
    print("\n✅ Normal retriever ativado: fragmentação sem hierarquia criada.")


class PlansTool(BaseTool):
    name: str = "plans_tool"
    description: str = (
        "Use this tool to read a JSON file containing information about plan prices and features."
    )

    file_path: str = None  # The file path is now expected to be passed as a parameter.

    def _run(self, file_path: str):
        """
        Reads the JSON file specified by the file_path and returns its contents.
        """
        self.file_path = file_path  # Set the file path dynamically
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # Load the JSON content
            return data
        except FileNotFoundError:
            return f"Error: File not found at {self.file_path}."
        except json.JSONDecodeError:
            return f"Error: Failed to decode JSON file at {self.file_path}."
        except Exception as e:
            return f"An unexpected error occurred: {e}"

    def _arun(self, file_path: str):
        """
        Asynchronous execution is not supported.
        """
        raise NotImplementedError("This tool does not support asynchronous execution.")
        

# Function: Decide next node based on if the tool os necessary or not
def decide_next_node_context(state: State) -> Literal["use_RAG", "plans_comparison_tool"]:
    ##print(state["context_decision"])
    if "plans_tool" in state["context_decision"]:
        return "plans_comparison_tool"        
    else:        
        return "use_RAG"

# Func: Verify lenght of context and make it short if necessary
def decide_next_node_lenght(state: State) -> Literal["make_history_short", END]:
    if len(state["context_chunck_history"]) > state["maximum_context"] or len(state["thread_states"][state["thread"]]) > state["maximum_dialogs"]:      
        return "make_history_short"
    else:
        return END

# Func: creates the system message
def system_message_RAG(context_list) -> str:    
   
    # Concatenar os valores da lista em uma única string, separando por um espaço ou outra separação desejada
    final_context = " ".join(filter(None, context_list))  # Remove valores None antes de concatenar    

    system_message_and_context = (
    "Continue the conversation strictly based on the provided context below.\n"
    "Context:\n"
    f"<context>\n{final_context}\n</context>\n\n"

    "### Instructions for answering:\n"
    "- Provide a concise, accurate, and direct answer based strictly on the given context.\n"   
    "- Do not include disclaimers, introduction or add unnecessary formatting.\n"

    "### Handling Image Paths and Descriptions:\n"
    "- The provided context contains multiple image paths, each immediately followed by a description explaining the content of that specific image.\n"
    "- If there is one or more images whose description explicitly and clearly supports or illustrates the topic, include the exact image paths immediately after your explanation of that topic.\n"
    "- Include **only** images relevant to each topic discussed, based on its description.\n"
    "- If no images explicitly support a topic, do not include any image paths and don't say anything about images.\n\n"
    "- Do not inlcude images descriptions in your answer, just use it to make your decisions about which images to include.n\n"

    "### Example context structure when image is relevant:\n"
    "[IMAGE: path/to/image.jpg]\n"
    "Description: This image shows a clear example of XYZ related to the answer.\n\n"

    "### Example Answer format when image is relevant:\n"
    "Clear and concise explanation of topic XYZ based strictly on the provided context.\n"
    "[IMAGE: path/to/image.jpg]\n\n"

    "### Example Answer format in case NO image is relevant:\n"    
    "Clear and concise explanation of topic XYZ based strictly on the provided context.\n"
    "In this case, do NOT mention anything about images and do NOT include any path"
       
)


    
    return system_message_and_context

# Func: create the system message for the model
def system_message_plans(context) -> str:
    system_message_and_context = (
        "Check if the user's question can be answered using only the content below."
        "The context represents a table comparing features across two subscription plans offered by Pixsee Planet."
        "If the answer is not in the context, reply with exactly: 'Sorry, I don't have this information.'."
        "If the answer is in the context, reply with **only** the exact answer, without any introduction, explanation, or additional text."
        "Do **not** repeat the question, do **not** rephrase, and do **not** include any disclaimers or formatting instructions."
        "Reply only with the necessary information and do not include text like 'Based on the provided context...'"
        
        f"<context>{context}</context>"
    )
    return system_message_and_context



def load_data(file_path: str) -> List[Dict]:
    """
    Carrega os dados estruturados do arquivo JSON.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def decide_next_node_validation(state: State) -> Literal["use_RAG", "check_history_lenght"]:
    """
    Validate if the tools's answer knows the answer, otherwise, try RAG.
    """
    ##print(state["llm_response"])
    if "i don't know" in state["llm_response"]["content"].lower() and state["context_decision"] == "plans_tool":           
        return "use_RAG"
    
    else:
        #update the threads with the llm-OUTput if the answer is valid
        state["thread_states"][state["thread"]].append({"role": "model", "content": state["llm_response"]})
        return "check_history_lenght"

#########################################################################################################

def use_RAG(state: State) -> State:
 
    state.update(dict.fromkeys(["graph_state", "final_decision", "context_decision"], "use_RAG"))

    return state


def context_decision_node(state):
    """
    Use if statement to decide which tool to use.
    """
    query = state["query"]
    
    # terms to test, considering variations, typos, etc.
    terms = ["xyz10017654"]  # a random string just to avoid this tool being called. 
    
    """terms = ["plan", 
             "subscription", 
             "subscribed", 
             "subscribing", 
             "subcription", 
             "subsciption", 
             "subscrption", 
             "enroll", 
             "enrolled", 
             "enrolling", 
             "enrollment"]"""

    decision = "use_RAG"
    for term in terms:
        if term.lower() in query.lower():
            decision = "plan_tools"
            break
    
    # Stores the decision in the state
    state["context_decision"] = decision

    return state

################ LOAD EMBEDDINGS ##############################
import os
import json
import uuid
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

@phoenix_trace_function(span_name="emb_simil", kind="EMBEDDING", embeddings_key="resulting_chunk_context", input_key="query", output_key="context_chunck_history")
def verify_embeddings_similarity_on_FAISS(state: State) -> State:
    import numpy as np
    #print("---verify_embeddings_similarity---")
    state["graph_state"] = "verify_embeddings_similarity_on_FAISS"

    #print(f"Using parent retriever: {use_parent_retriever}")
             
    k_embeddings = state["top_k"]
    #print(f"top_k: {k_embeddings}")

    if state["modify_query"] == False:
        query_embedding = ml_client.get_embedding(state["query"])['embedding']
    else:
        query_embedding = ml_client.get_embedding(state["modified_query"])['embedding']

    query_embedding = np.array(query_embedding).astype("float32")
    
    # 🔹 3. Normalizar o embedding (importante para FAISS L2)
    query_embedding = np.array(query_embedding).astype("float32")
    #query_embedding /= np.linalg.norm(query_embedding)  # Normalização L2

    faiss_index = vectorstore.index
    
    # Realizar a busca FAISS em baixo nível
    num_vectors = faiss_index.ntotal  # Obtém o número total de vetores indexados
    distances, indices = faiss_index.search(query_embedding.reshape(1, -1), num_vectors)

    #only for debug
    chunks_fragments_found = [vectorstore.docstore._dict[vectorstore.index_to_docstore_id[i]] for i in indices[0] if i != -1]
  

    if state["use_parent_retriever"] == True:
        # 🔹 9. Buscar os documentos pais correspondentes aos fragmentos filtrados
        selected_parent_chunk_ids = set()  # Conjunto para armazenar IDs únicos
        retrieved_docs = []  # Lista para armazenar documentos pais
        
        for fragment in chunks_fragments_found:
            parent_chunk_id = str(fragment.metadata["parent_chunk_id"])            
            # Adiciona apenas se for um novo documento pai
            if parent_chunk_id not in selected_parent_chunk_ids:
                selected_parent_chunk_ids.add(parent_chunk_id)
                retrieved_doc = parent_store.mget([parent_chunk_id])[0]  # Buscar o documento pai
        
                if retrieved_doc is not None:
                    retrieved_docs.append(retrieved_doc)
        
  
      # 🔹 Exibir os documentos pais recuperados
        ##print("\n📄 Documentos pais recuperados:")
        if retrieved_docs:
            for doc in retrieved_docs:                
                ##print(f"📌 Metadata: {doc.page_content} | Documento: {doc.metadata}\n")
                pass
                
        else:
            print("⚠︝ Nenhum documento pai foi recuperado.")         
   
    else:  # Do not sue parent retriever 
        print("Not using parent retriever")
        
        # 🔹 Recuperar os documentos correspondentes com base nos índices encontrados
        retrieved_docs = [vectorstore.docstore._dict[vectorstore.index_to_docstore_id[i]] for i in indices[0] if i != -1]
        
    final_docs = retrieved_docs
    

    #print(f"filter by company: {state['filter_by_company']}")
    # if filter by company name or date is selected
    if state["filter_by_company"]:           
        
        # 🔹 2. Preparar a query e calcular o embedding
        from sentence_transformers import SentenceTransformer
        import numpy as np         

        
        # Obter a lista de empresas para filtro e converter para minúsculas
        company_filter_list = [company.lower() for company in state.get("extracted_metadata", {}).get("company_name", [])]
              
        # Inicializar listas de documentos filtrados e pontuações
        filtered_docs = []
        #filtered_scores = []
        
        # 🔹 1. Inicializar final_docs como vazio
        filtered_docs = []
        
        # 🔹 2. Se houver nomes de empresa, aplicar o filtro
        if company_filter_list:
            #print("will filter")
            for doc in final_docs:         
                company_name = doc.metadata.get("company_name", "Desconhecido")  # Evita erro caso não exista
                
                # 🔹 Verifica se o "company_name" contém qualquer empresa da lista           
                if any(target in company_name.lower() for target in company_filter_list):
                    filtered_docs.append(doc)
  
            if filtered_docs:
                final_docs = filtered_docs            
      
    #print(f"filter by date: {state['filter_by_date']}")    

    
    #limit documents to top_k
    final_docs = final_docs[:k_embeddings]
    
    # 🔹 Exibir os documentos  recuperados
    ##print("\n📄 Documentos recuperados após filtragem (ou não) e limit top_k:")
    if final_docs:
        for doc in final_docs:
            pass
            ##print(f"📌 Metadata: {doc.page_content} | Documento: {doc.metadata}\n")
    else:
        #print("not using text similarity")
        final_docs = retrieved_docs

        
    # Converter lista de documentos em uma lista de dicionários
    documents_dict_list = final_docs 
    #[
        #{**doc.metadata, "text": doc.page_content} for doc in final_docs
    #]
 
          
    # 🔹 Atualiza o estado do sistema
    #state["doc_texts"] = ["\n".join(d["content"] for d in documents_dict_list)]
    state["doc_texts"] = ["\n".join(f"{d['content']}" for d in documents_dict_list)]


    state["resulting_chunk_context"] = documents_dict_list

    if len(retrieved_docs) == 0:
        print("No metadata returned")
    
    # 🔹 Estender o histórico de contexto
    state["context_chunck_history"].extend(state["resulting_chunk_context"])
    
    # 🔹 Limpar conteúdo desnecessário
    state["query_embedding"] = None
    

    return state

        
   

# Node: Build system message
def create_system_message(state: State) -> State:
    #print("---create_system_message---")
    
    if state["graph_state"] == "plans_comparison_tool":
        state["system_message"] = system_message_plans(state["doc_texts"])
        #print("call tool function")

    else:
        #print("Tool decision: call RAG function")
        #if the call is for the RAG, bring the context history
        state["system_message"] = system_message_RAG(state["doc_texts"]) 
          
                 
    state["graph_state"] = "create_system_message"
    return state


# Node: build query_template
@phoenix_trace_function(span_name="prompt", kind="LLM", input_key="query", output_key="query_template")
def build_prompt_template(state: State) -> State:
    #print("---build_prompt_template---")
    state["graph_state"] = "build_prompt_template"    

    if state["thread"] in state["thread_states"]:      
        
        # Add the query to the thread
        state["thread_states"][state["thread"]].append({"role": "user", "content": state["query"]}) 

        template = [{"role": "system", "content": state["system_message"]}]
       
        template.extend(state["thread_states"][state["thread"]])

        state["query_template"] = copy.deepcopy(template)       
            
    else:
        #create initial prompt template        
        state["query_template"] = copy.deepcopy([{"role": "system", "content": state["system_message"]},
                        {'role': 'assistant', 'content': 'Hi, Please ask any question about your documents.'},                        
                        {"role": "user", "content": state["query"]}])

        # create threads for the first time
        state["thread_states"][state["thread"]] = copy.deepcopy([{"role": "user", "content": state["query"]}])

        
  
    return state

def modify_query_node(state: State) -> State:
    #print("---node--modify_query---")
    state["graph_state"] = "modify_query"    
    # Generate embeddings for the user's query  

    pdf_indice = state["pdf_indice"]

    # Lê o conteúdo do arquivo
    with open(state["pdf_indice"], "r", encoding="utf-8") as f:
        pdf_indice_content = f.read()

   
    if state["modify_query"] == True: # Query expansion
        if state.get("query_template"):            
            modified_query = state["query_template"][1:]
        else:
            modified_query = [{'role': 'assistant', 'content': 'Hi, Please ask me any question about the content of your documents.'}, {'role': 'user', 'content': state['query']}]             
        
        prompt = [
            {
                "role": "user",
                "content": f"""
        Your task is to generate a phrase composed of the most relevant user query, followed by ':', followed by the best-fitting section(s) from the list below.

        ### Important:
        - If the last user query is a follow-up query, identify the most relevant previous query instead.
        - If multiple sections are relevant, return all applicable sections, separated by commas.
        - Respond **only** with the relevant query and the relevant section(s). No introductions or explanations.
         
        
        ## Conversation: \"{modified_query}\"
        
        ### Sections:
        {pdf_indice_content}           
        
        ### Answer Format:
        <Query>: <Section(s)>
        
        ### Examples:
        I have a flat tire, what should I do?: Maintenance (Tire Care and Maintenance), Maintenance (Temporary Tire Repair Kit)
        How do I clean my car?: Maintenance (Cleaning)
        How do I use Autopilot?: Autopilot (Autopilot Features)
        How to open the window?: Opening and Closing (Windows)
        """
            }
        ]

        outputs = ml_client.llm_generate(prompt)
        final_query = outputs["content"][-1] if 'content' in outputs else outputs
                
        state["modified_query"] = state["query"] + ": " + final_query["content"]

    else:
         state["modify_query"] = False
    

    return state

@phoenix_trace_function(span_name="llm_call", kind="LLM", input_key="query", output_key="llm_response.content")
def llm_call(state: State) -> State:     
    
    #print("---llm_call---")    
    state["graph_state"] = "llm_call"
   
    prompt = state["query_template"]
    outputs = ml_client.llm_generate(prompt)
  
    answer = outputs["content"][-1] if 'content' in outputs else outputs
    #answer = {"content": "llm turned off"}  # simulation for debugging 

    state["llm_response"] = answer  
    
    
    return state


def make_history_short(state: State) -> State:
    #print("---make_history_short---")    
    state["graph_state"] = "make_history_short"  
  

    # Limit the history of dialogs and context     
    state["thread_states"][state["thread"]] = state["thread_states"][state["thread"]][-(state["maximum_dialogs"]*2):]
    state["context_chunck_history"] = state["context_chunck_history"][-(state["maximum_context"]):]    

    return state

def bring_conversation_history(state: State) -> State:
    state["graph_state"] = "bring_conversation_history" 
    return state



def plans_comparison_tool(state):
    """
    Process the query using plans tool
    """
    state["graph_state"] = "plans_comparison_tool"

    current_directory = os.getcwd()
    path = os.path.join(current_directory, "structured_subscription_data_with_categories.json")
    state["resulting_context"] = PlansTool()._run(path)
    state["final_decision"] = "plan_tools"
    return state
    
def validate_answer(state):
    """
    Validate wether the tools have the answer
    """
    return state
        
# Node: Receive the query
def check_history_lenght(state: State) -> State:
    #print("---check_history_lenght---")    
    state["graph_state"] = "check_history_lenght"
    return state


