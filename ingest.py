import os
from concurrent.futures import ThreadPoolExecutor
from typing import List
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.openai import OpenAIEmbedding
from elasticsearch import Elasticsearch, NotFoundError
from dotenv import load_dotenv

load_dotenv()

embed_model = OpenAIEmbedding(embed_model="text-embedding-3-small", api_key=os.getenv("OPENAI_API_KEY"))

node_parser = SimpleNodeParser().from_defaults(
    chunk_size=1024,
    chunk_overlap=50
)

es_client = Elasticsearch("http://localhost:9200")

storage_context = StorageContext.from_defaults(
    vector_store=ElasticsearchStore(
        es_url="http://localhost:9200",
        dim=1536,
        index_name="index_2025"
    )
)  

try:
    es_client.indices.delete(index=storage_context.vector_store.index_name)
except NotFoundError:
    pass

def process_pdfs(directory_path):

    reader = SimpleDirectoryReader(
        input_dir=directory_path,
        recursive=True,
        required_exts=[".pdf"],
        filename_as_id=True
    )    
    documents = reader.load_data()
 
    for doc in documents:

        nodes = node_parser.get_nodes_from_documents([doc], show_progress=True)
        texts = [n.text for n in nodes]
        embeddings = embed_model.get_text_embedding_batch(show_progress=True, texts=texts)
        nodes = [
            TextNode(
                text=text, 
                embedding=embedding, 
                metadata={
                    "filename": doc.extra_info['file_path'], 
                    "page_no": doc.extra_info['page_label']
                }
            )
            for text, embedding in zip(texts, embeddings)
        ]        
        
        storage_context.vector_store.add(nodes)
        
        print(f"Processed and indexed: {doc.extra_info['file_path']}")

if __name__ == "__main__":
    # Specify your PDF directory path
    pdf_directory = "./docs"
    
    # Process all PDFs in directory
    process_pdfs(pdf_directory)
    VectorStoreIndex.from_vector_store(storage_context.vector_store).storage_context.vector_store.close()

    print("Indexing complete!")
