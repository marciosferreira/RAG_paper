import os
import shutil
from pathlib import Path
from processor.config import Config  # Suponho que Config seja a classe para carregar o YAML
from processor.pdf_processor import PDFProcessor
from processor.chunker import ChunkProcessor
from processor.images_description import MarkdownImageDescriber
from processor.faiss_indexer import FAISSIndexer
from huggingface_hub import snapshot_download
import sys
from pathlib import Path
import json
from langchain.schema import Document
import json


def main():
    # Caminho para o config.yaml
    this_dir = Path(__file__).parent
    config_path = this_dir / "config.yaml"  # Carregando do config.yaml

    # Inicializa a configuração e a classe de descritora de imagem
    config = Config(config_path)
    describer = MarkdownImageDescriber(config)

    # Caminho para os PDFs
    pdf_dir = config.get("pdf_path")
    output_base_dir = Path(config.get("output_dir"))

    # Obter as pastas de output já existentes
    existing_output_dirs = {f.stem for f in output_base_dir.iterdir() if f.is_dir()}

    print("# Checking for new PDFs to process...")

    # Itera sobre os arquivos PDF
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            pdf_name = os.path.splitext(filename)[0]  # nome do PDF sem extensão

            # Verifica se já existe uma pasta correspondente ao PDF
            output_dir = output_base_dir / pdf_name

            # Se a pasta já existe, pula o processamento para esse PDF
            if pdf_name in existing_output_dirs:
                print(f"# PDF '{filename}' already processed, skipping...")
                continue  # Pula para o próximo arquivo

            # Caso contrário, processa o PDF (Step 1)
            print(f"# Processing new PDF: {filename}")
            processor = PDFProcessor(pdf_path, output_dir, config.get("markdown_file_name"))
            markdown_path, markdown_file = processor.convert_pdf_to_markdown()
            print(f"### -> Processed: {filename} → {markdown_path}")

            # Agora executa a descrição das imagens para esse novo PDF
            print(f"# Running image descriptor on Markdown: {markdown_path}")
            describer.process_markdown(markdown_path, output_dir)  # Passando o caminho do Markdown e o diretório de saída

    print("# Running further steps for all folders (new and existing)...")

    # Continue com os outros passos, como o processamento de chunks (Step 3)
    # e os outros passos que você já tem no código.

    print("# Pipeline finished successfully!")
    
    print("# Processing the chunks in each output folder...")

    output_dir = Path(config.get("output_dir"))
    markdown_file_name = Path(config.get("markdown_file_name"))
    index_filename = config.get("name_tosave_index")

    
    for subfolder in output_dir.iterdir():  
        if not subfolder.is_dir():
            print('one folder slkip')
            
            continue
    
        markdown_path = subfolder / markdown_file_name
    
        if not markdown_path.exists():
            print(f"### Markdown file not found in: {subfolder}")
            continue
    
        print(f"### Processing chunks for: {markdown_path.name}")
    
        # Define os caminhos de saída para JSONs dentro da pasta do PDF
        parent_json_path = subfolder / "parent.json"
        child_json_path = subfolder / "child.json"
        path_tosave_indice = subfolder / index_filename
    
        chunk_processor = ChunkProcessor(
            markdown_path,
            parent_json_path,
            child_json_path,
            path_tosave_indice
        )
    
        child_chunks = chunk_processor.process()
    
        if child_chunks:
            print(f"### {len(child_chunks)} chunks created for: {markdown_path.name}")
        else:
            print(f"### No chunks generated for: {markdown_path.name}")

####################################################################

    print("# FAISS indexing...")
    faiss_indexer = FAISSIndexer(
        config.get("faiss_index_path"),
        config.get("model_embeddings_path")
    )
    faiss_indexer.create_faiss_index(child_chunks)
####################################################################

    
    # Lista para armazenar os chunks de todos os parent.json
    all_parent_chunks = []
    
    output_dir = Path(config.get("output_dir"))
    print("resulting_parent_json_path:", config.get("resulting_parent_json_path"))

    resulting_parent_json_path = Path(config.get("resulting_parent_json_path"))
    
    print("# Coletando todos os parent_chunks...")
    
    # Itera pelas subpastas dentro de output_dir
    for subfolder in output_dir.iterdir():
        if not subfolder.is_dir():
            continue
    
        parent_json_path = subfolder / "parent.json"
        if not parent_json_path.exists():
            print(f" Arquivo parent.json não encontrado em: {subfolder}")
            continue
    
        # Carrega o conteúdo do parent.json da subpasta
        with open(parent_json_path, "r", encoding="utf-8") as f:
            parent_chunks = json.load(f)
    
        # Adiciona os chunks carregados à lista geral
        all_parent_chunks.extend(parent_chunks)
    
    print(f"# Total de parent_chunks coletados: {len(all_parent_chunks)}")
    
    # Caminho do diretório onde o arquivo será salvo
    resulting_parent_json_path = Path(config.get("resulting_parent_json_path"))
    
    # Crie o diretório, se necessário
    resulting_parent_json_path.mkdir(parents=True, exist_ok=True)
    
    # Defina o caminho completo para o arquivo final (geral_parent.json)
    final_file_path = resulting_parent_json_path / "geral_parent.json"
    
    # Salva o JSON final concatenado no local indicado por final_file_path
    with open(final_file_path, "w", encoding="utf-8") as f:
        json.dump(all_parent_chunks, f, indent=4, ensure_ascii=False)
    
    print(f"# Arquivo final de parent_chunks salvo em: {final_file_path}")

#############################################################


   
    # Lista para armazenar os chunks de todos os child.json
    all_child_chunks = []
    
    # Diretório onde os arquivos child.json estão localizados
    output_dir = Path(config.get("output_dir"))
    print("resulting_child_json_path:", config.get("resulting_child_json_path"))
    
    # Caminho onde o arquivo final será salvo
    resulting_child_json_path = Path(config.get("resulting_child_json_path"))
    
    print("# Coletando todos os child_chunks...")
    
    # Itera pelas subpastas dentro de output_dir
    for subfolder in output_dir.iterdir():
        if not subfolder.is_dir():
            continue
    
        child_json_path = subfolder / "child.json"
        if not child_json_path.exists():
            print(f" Arquivo child.json não encontrado em: {subfolder}")
            continue
    
        # Carrega o conteúdo do child.json da subpasta
        with open(child_json_path, "r", encoding="utf-8") as f:
            child_chunks = json.load(f)
    
        # Adiciona os chunks carregados à lista geral
        all_child_chunks.extend(child_chunks)
    
    print(f"# Total de child_chunks coletados: {len(all_child_chunks)}")
    
    # Cria o diretório onde o arquivo final será salvo, se necessário
    resulting_child_json_path.mkdir(parents=True, exist_ok=True)
    
    # Defina o caminho completo para o arquivo final (general_child.json)
    final_child_file_path = resulting_child_json_path / "general_child.json"
    
    # Salva o JSON final concatenado no local indicado por final_child_file_path
    with open(final_child_file_path, "w", encoding="utf-8") as f:
        json.dump(all_child_chunks, f, indent=4, ensure_ascii=False)
    
    print(f"# Arquivo final de child_chunks salvo em: {final_child_file_path}")


###################################################################

    
    # Caminho onde o arquivo final será salvo
    path_tosave_index = Path(config.get("path_tosave_index"))
    path_tosave_index.mkdir(parents=True, exist_ok=True)  # Cria o diretório, se não existir
    
    final_index_path = path_tosave_index / "general_index.txt"
    
    # Lista para armazenar o conteúdo de todos os pdf_indice.txt
    all_pdf_indices = []
    
    print("# Coletando todos os pdf_indice.txt...")
    
    # Itera pelas subpastas dentro de output_dir
    for subfolder in output_dir.iterdir():
        if not subfolder.is_dir():
            continue
    
        pdf_indice_path = subfolder / "pdf_indice.txt"
        if not pdf_indice_path.exists():
            print(f" Arquivo pdf_indice.txt não encontrado em: {subfolder}")
            continue
    
        # Carrega o conteúdo do pdf_indice.txt da subpasta
        with open(pdf_indice_path, "r", encoding="utf-8") as f:
            pdf_indice_content = f.read()
    
        # Adiciona o conteúdo à lista
        all_pdf_indices.append(pdf_indice_content)
    
    print(f"# Total de arquivos pdf_indice.txt coletados: {len(all_pdf_indices)}")
    
    # Salva o conteúdo concatenado no arquivo final
    with open(final_index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_pdf_indices))
    
    print(f"# Arquivo final de índice salvo em: {final_index_path}")
    
############################################################################################

    print("# Pipeline sucessfull finished!")


if __name__ == "__main__":
    main()

