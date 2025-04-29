###### use this version to use the llama vision model 
           
import re
from pathlib import Path
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

class MarkdownImageDescriber:
    def __init__(self, config):
        self.config = config
        self.markdown_file_name = Path(config.get("markdown_file_name"))
        self.output_dir = Path(config.get("output_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_pattern = re.compile(r"!\[(.*?)\]\((.*?)\)")

        self.model_id = config.get("model_vision_repo_id")
        self.model_path = config.get("model_vision_path")

        print("# Downloading Vision Model (if needed)...")
        snapshot_download(repo_id=self.model_id, local_dir=self.model_path)

        print("# Loading Vision Model and Processor...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
        )

        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=bnb_config,
        )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def describe_image(self, image_path, prompt_text=None):
        image = Image.open(image_path)
        if prompt_text is None:
            prompt_text = "Please describe the image briefly."

        input_text = f"<|image|><|begin_of_text|>{prompt_text}"

        inputs = self.processor(image, input_text, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=128)

        image.close()
        response = self.processor.decode(output[0], skip_special_tokens=True)
        if "." in response:
            response = response.split(".", 1)[-1].strip()
        return response

    def process_markdown(self):
        print("# Starting batch Markdown processing...")

        for subfolder in self.output_dir.iterdir():
            if subfolder.is_dir():
                markdown_path = subfolder / self.markdown_file_name

                if not markdown_path.exists():
                    print(f"# Markdown file not found in: {subfolder}")
                    continue

                print(f"# Processing: {markdown_path}")

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

                print(f"# ✅ Updated: {markdown_path}")
