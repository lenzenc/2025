import os
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.openai import OpenAIEmbedding
from openai import OpenAI
from llama_index.core.retrievers import VectorIndexRetriever
from dotenv import load_dotenv

load_dotenv()

# Initialize embedding model
embed_model = OpenAIEmbedding(
    embed_model="text-embedding-3-small", 
    api_key=os.getenv("OPENAI_API_KEY")
)
Settings.embed_model = embed_model

openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

storage_context = StorageContext.from_defaults(
    vector_store=ElasticsearchStore(
        es_url="http://localhost:9200",
        dim=1536,
        index_name="index_2025"
    )
)  

index = VectorStoreIndex.from_vector_store(
    storage_context.vector_store
)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10
)

def query_documents(query_text):

    results = retriever.retrieve(query_text)    
    context = '\n'.join([result.node.text for result in results])

    prompt = f"""
        You are an expert when it comes to providing the best quality answer given an question from an user and provided context.
        When possible please only use the context data to help anser the users question.

        Context:
        {context}

        User Question: {query_text}
    """
        
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": prompt
        }]
    )
    print(resp.choices[0].message.content)

if __name__ == "__main__":

    query = input("Ask a question:")
    query_documents(query)
    index.storage_context.vector_store.close()
