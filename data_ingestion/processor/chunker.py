import re
import uuid
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os


class ChunkProcessor:
    def __init__(self, markdown_path, parent_json_path, child_json_path, path_tosave_indice):
        self.markdown_path = markdown_path
        self.parent_json_path = parent_json_path
        self.child_json_path = child_json_path
        self.path_tosave_indice = path_tosave_indice

        self.image_pattern = re.compile(r"\[IMAGE: .*?\]")
        self.markdown_image_pattern = re.compile(r"!\[.*?\]\((.*?)\)")
      
    def extract_parent_chunks(self, md_text):
        parent_chunks = []
        titles_for_txt = []
        current_chunk = None
        lines = md_text.split("\n")

        for line in lines:
            if line.startswith("## "):
                clean_line = line.lstrip("#").strip()
                if len(clean_line) > 1:
                    titles_for_txt.append(clean_line)
                if current_chunk:
                    parent_chunks.append(current_chunk)
                current_chunk = {"title": line.strip(), "content": line.strip() + "\n\n", "images": [], "chunk_id": str(uuid.uuid4())}
            elif current_chunk:
                image_match = self.markdown_image_pattern.findall(line)
                if image_match:
                    for img_path in image_match:
                        current_chunk["images"].append(img_path)
                        current_chunk["content"] += f"[IMAGE: {img_path}]\n"
                else:
                    current_chunk["content"] += line + "\n"

        if current_chunk:
            parent_chunks.append(current_chunk)
            

        # Salva os títulos em formato compacto com delimitador " || "
        with open(self.path_tosave_indice, "w", encoding="utf-8") as f:
            f.write(" || ".join(titles_for_txt))

        return parent_chunks

    def create_child_chunks(self, parent_chunks):
        child_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50, separators=["\n\n", "\n"])

        for parent in parent_chunks:
            content_cleaned = re.sub(self.image_pattern, "", parent["content"]).strip()
            fragments = splitter.split_text(content_cleaned)

            for frag in fragments:
                child_chunks.append(Document(
                    page_content=f"{parent['title']}\n\n{frag.strip()}",
                    metadata={"parent_chunk_id": parent["chunk_id"], "title": parent["title"]}
                ))
        
        return child_chunks

    def process(self):
        print("### Gerando novos parent_chunks...")
        with open(self.markdown_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()
        
        parent_chunks = self.extract_parent_chunks(markdown_text)
        
        with open(self.parent_json_path, "w", encoding="utf-8") as f:
            json.dump(parent_chunks, f, indent=4, ensure_ascii=False)        
    
        # Os child_chunks sempre são regenerados a partir dos parent_chunks já carregados
        child_chunks = self.create_child_chunks(parent_chunks)
    
        with open(self.child_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"content": doc.page_content, "metadata": doc.metadata} for doc in child_chunks],
                f,
                indent=4,
                ensure_ascii=False
            )
    
        print("# Current Working Directory:", os.getcwd())
        print("# process() concluído com sucesso.")
    
        return child_chunks

