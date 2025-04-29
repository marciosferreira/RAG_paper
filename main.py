"""
This is a CLI interface to use the system,
allow us to: load new documents, make query in the RAG
"""

import os
import argparse
import logging
from pathlib import Path
# from data_ingestion.embedding import TextEmbeddingPipeline
# from data_ingestion.vector_db import VectorDB
from src import graph_builder
# from src import ingestion
# from retrieval.retriever import Retriever
# from generation.generator import Generator

# Initialize components
# vector_db = VectorDB()
# retriever = Retriever(vector_db)
# generator = Generator()

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def process_documents(pdf_file_path):
    """Load and generate embeddings."""
    if os.path.splitext(pdf_file_path)[-1] == '.json':
        logger.info("starting from Json")
        import json
        with open(pdf_file_path) as json_data:
            processed_chunks = json.load(json_data)
    else:
        output_dir = Path(__file__).absolute().parent / '../scratch'
        pipeline = TextEmbeddingPipeline(Path(pdf_file_path), output_dir)

        processed_chunks = pipeline.run()

    # vector_db.add_documents(processed_chunks)

def query_rag(question, mode):
    """Retrieve relevant docs and generate an answer."""
    graph_builder.main(mode)
    # docs = retriever.get_relevant_docs(question)
    # context = "\n".join([doc["content"][:500] for doc in docs])
    # response = generator.generate_response(question, context)
    # print("\n📖 **Answer:**", response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG system")
    parser.add_argument("--doc", type=str, help="A pdf file to be loaded")
    parser.add_argument("--query", action="store_true", default=False, help="Ask a question to the RAG system")
    parser.add_argument("--audio", action="store_true", default=False, help="Ask a question to the RAG system")
    ### TEMPORARY ####
    parser.add_argument("--tesla", action="store_true", help="just to make fast loading tesla dataset")
    ########

    args = parser.parse_args()

    
    if args.doc or args.tesla:
        raise NotImplementedError('if you want to precess a new document refer to the README to see how it is done in this version')
        if args.tesla:
            input_doc_path = "outputs/processed_tesla.json"
            # ingestion.main.main()
        else:
            process_documents(args.doc)
    if args.query:
        query_rag(args.query, 'query')
    elif args.audio:
        query_rag(args.query, 'audio')
    

    if not (args.doc or args.tesla) and not args.query:
        print("Please provide either --docs <folder_path> or --query <question>")
