# config.py - Configuração e Inicialização
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

# Carregar configurações do YAML
config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
with open(config_path, "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

root_path = Path(config.get("project_path", "."))

# Definir variáveis a partir do YAML
model_id = config["weights"]["id"]
llm_model_path = str((root_path / config["weights"]["llm_model_path"]).resolve())
model_embeddings_path = str((root_path / config["weights"]["embeddings_path"]).resolve())

markdown_file = str((root_path / config["paths"]["markdown_file"]).resolve())
parent_json_path = str((root_path / config["paths"]["parent_json"]).resolve())
child_json_path = str((root_path / config["paths"]["child_json"]).resolve())
faiss_index_path = str((root_path / config["paths"]["faiss_index"]).resolve())
faiss_index_path_config = str((root_path / config["paths"]["faiss_index_path_config"]).resolve())
pdf_indice = str((root_path / config["paths"]["pdf_indice"]).resolve())


use_parent_retriever = config["retriever"]["use_parent"]
use_cossine_similarity = config["embeddings"]["use_cosine_similarity"]
tracing_using_phoenix_arize = config["tracing"]["phoenix_arize"]