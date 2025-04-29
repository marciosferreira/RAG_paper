import re
from pathlib import Path
from PIL import Image
from transformers import BlipForConditionalGeneration, MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
import sys


class MarkdownImageDescriber:
    def __init__(self, config):
        self.config = config
        self.markdown_file_name = Path(config.get("markdown_file_name"))
        self.output_base_dir = Path(config.get("output_dir"))
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")
        self.model_id = config.get("model_vision_repo_id")
        self.model_path = config.get("model_vision_path")

        print("# Loading Vision Model and Processor...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )
    
        if self.model_id == "Salesforce/blip-image-captioning-base":

            print("# Downloading Vision Model Salesforce/blip-image-captioning-base if needed...")
            snapshot_download(repo_id=self.model_id, local_dir=self.model_path)              
    
            self.model = BlipForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                quantization_config=bnb_config,
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)

        
        elif self.model_id =="meta-llama/Llama-3.2-11B-Vision-Instruct":
            
            print("# Downloading Vision Model meta-llama/Llama-3.2-11B-Vision-Instruct if needed...")
            snapshot_download(repo_id=self.model_id, local_dir=self.model_path)      

            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=bnb_config,
            )
            
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            
        else:
            print("## Not valid model for this version...exiting")
            sys.exit()

    def describe_image(self, image_path, prompt_text=None):

        if self.model_id == "Salesforce/blip-image-captioning-base":

            image = Image.open(image_path).convert("RGB")  # BLIP requer RGB
        
            if prompt_text is None:
                prompt_text = "a photo of"  # prompt padrão para BLIP
        
            # Pré-processa a imagem e o prompt
            inputs = self.processor(image, prompt_text, return_tensors="pt").to(self.model.device)
        
            # Gera a legenda
            output = self.model.generate(**inputs, max_new_tokens=64)
        
            # Decodifica a saída
            description = self.processor.decode(output[0], skip_special_tokens=True)
        
            image.close()
            return description
            
        else:
            image = Image.open(image_path)
            if prompt_text is None:
                prompt_text = "Please describe the image briefly."
    
            input_text = f"<|image|><|begin_of_text|>{prompt_text}"
    
            inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=128)
    
            image.close()
            response = self.processor.decode(output[0], skip_special_tokens=True)
            if "." in response:
                description = response.split(".", 1)[-1].strip()
            return description

            

    def process_markdown(self, markdown_path, output_dir):
        print(f"# Processing Markdown: {markdown_path}")
        
        with open(markdown_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()

        matches = self.image_pattern.findall(markdown_text)
        print(f"# Found {len(matches)} image(s) in {markdown_path.name}")

        for alt_text, image_url in tqdm(matches, desc=f"# Describing images in {markdown_path.name}"):
            image_path = Path(image_url)
            if not image_path.is_absolute():
                image_path = markdown_path.parent / image_path

            if image_path.exists():
                description = self.describe_image(image_path)
                image_syntax = f"![{alt_text}]({image_url})"
                description_text = f"{image_syntax}\n_Image Description: {description}_"
                markdown_text = markdown_text.replace(image_syntax, description_text)
            else:
                print(f"# Image not found: {image_path}")

        # Sobrescreve o mesmo arquivo
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)

        print(f"### Updated: {markdown_path}")

    def process_new_pdfs(self, output_dir):
        # Processa apenas PDFs novos, ou seja, onde não existe uma pasta correspondente
        for subfolder in output_dir.iterdir():
            if subfolder.is_dir():
                markdown_path = subfolder / self.markdown_file_name

                if not markdown_path.exists():
                    print(f"# Markdown file not found in: {subfolder}")
                    continue

                # Chama a função de processamento apenas para novos PDFs
                print(f"# Processing new PDF Markdown: {markdown_path}")
                self.process_markdown(markdown_path, subfolder)

