from typing_extensions import TypedDict
from typing import List, Dict, Literal

class State(TypedDict):
    graph_state: str
    query: str
    embeddings_loaded: bool
    query_template: str
    query_embedding: List
    metadados: List
    resulting_chunk_context: List
    llm_response: str
    thread: str
    context_decision: str
    system_message: str
    context_chunck_history: List
    thread_states: Dict
    maximum_dialogs: int
    maximum_context: int
    final_decision: str
    adit_metadata: Dict
    embeddings_file_path: str
    full_docs_path: str
    doc_texts: List
    use_cossine_similarity: bool
    expanded_query: str
    extracted_metadata: List
    modify_query: bool
    modified_query: str
    top_k: int
    filter_by_company: bool
    filter_by_date: bool
    use_parent_retriever: bool
    pdf_indice: str