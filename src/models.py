"""
This is the inference engine
"""
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline

from langchain_huggingface import HuggingFaceEmbeddings


try:
    from config import model_embeddings_path, llm_model_path
except:
    from .config import model_embeddings_path, llm_model_path

def embedding_model(model_name=None):
    model_name = model_name if model_name is not None else model_embeddings_path
    embeddings_model = HuggingFaceEmbeddings(cache_folder=model_embeddings_path)
    return embeddings_model

class EmbeddingModel:
    def __init__(self, model_name=None):

        self.model_name = model_name if model_name is not None else model_embeddings_path
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)

    def __call__(self, text):
        return self.embedding_model.embed_query(text)

class LLMGenerator:
    
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path is not None else llm_model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.pad_token_id = (
            self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None
            else self.tokenizer.pad_token_id
        )

        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model_path,
            do_sample=False,
            torch_dtype=torch.float16,
            pad_token_id=self.pad_token_id,
            model_kwargs={
                "device_map": "auto",
                "quantization_config": self.quantization_config,
            },
        )


    def __call__(self, prompt, **kwargs) -> str:
        """
        Generates text from a prompt using the LLM pipeline.
        Additional generation arguments can be passed via kwargs.
        """
        max_new_tokens=kwargs.get('max_new_tokens', 500)
        do_sample=kwargs.get('do_sample', False)
        output = self.pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample)

        return output[0]["generated_text"]

