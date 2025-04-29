import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class FAISSIndexer:
    def __init__(self, faiss_index_path, model_embeddings_path):
        self.faiss_index_path = faiss_index_path
        self.embeddings_model = HuggingFaceEmbeddings(model_name=model_embeddings_path)

    def create_faiss_index(self, child_chunks):
        print("# Iniciando criação do índice FAISS...")
        print(f"# Número de chunks recebidos: {len(child_chunks)}")

        if len(child_chunks) == 0:
            raise ValueError("Nenhum chunk filho criado. FAISS não pode ser construído.")
        
        print(f"{len(child_chunks)} chunks filhos criados. Construindo índice FAISS do zero...")
        
        vectorstore = FAISS.from_documents(child_chunks, self.embeddings_model)
        print(f"# Índice FAISS criado: {vectorstore}")

        vectorstore.save_local(self.faiss_index_path)
        print(f"# Tentando salvar índice em: {self.faiss_index_path}")

        if os.path.exists(self.faiss_index_path):
            print(f"# Diretório de índice existe: {self.faiss_index_path}")
            print(f"# Arquivos presentes: {os.listdir(self.faiss_index_path)}")
        else:
            print("# Erro: Diretório de índice não foi criado!")

        print(f"# Caminho absoluto: {os.path.abspath(self.faiss_index_path)}")
        print(f"# Permissão de escrita? {os.access(self.faiss_index_path, os.W_OK)}")

        return vectorstore
