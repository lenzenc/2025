# 2025 Example
I don't support 2025 policy but it was an interesting document to create a RAG solution using LLama-Index, ElasticSearch and OpenAI

## Run
- mv .env.example .env and add your OPENAI_API_KEY
- pip install -r requirements.txt
- python ingest.py # ingests the document as embeddings to ES
- python ask.py # to ask a question related to the document