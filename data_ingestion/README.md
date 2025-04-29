The module data_ingestion structure

project/
│── main.py
│── config.json
│── processor/
│   ├── __init__.py
│   ├── config.py
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── faiss_indexer.py

1) Run the requirements.txt provided in main folder

2) The ingestion script must be run before the RAG script, as this is suposed to create the FAISS index.
   You just need to run it once, except if you want to change the pdf.

3) Insert all correct paths in: config.json
   
5) Run the pipeline: python main.py

