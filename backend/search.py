from elasticsearch import AsyncElasticsearch
import os
from typing import List, Optional
import json

# Global elasticsearch client
es_client: Optional[AsyncElasticsearch] = None

async def initialize_search():
    """Initialize Elasticsearch connection"""
    global es_client
    
    es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    es_client = AsyncElasticsearch(hosts=[es_url])
    
    try:
        # Test connection
        await es_client.ping()
        print(f"✅ Connected to Elasticsearch at {es_url}")
    except Exception as e:
        print(f"❌ Failed to connect to Elasticsearch: {e}")
        raise

async def find_relevant_docs(query: str, max_results: int = 5) -> List[str]:
    """Find relevant documents from the knowledge base"""
    if not es_client:
        return []
    
    try:
        # Simple text search (in production, use semantic/vector search)
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["content", "tags"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": max_results,
            "_source": ["content", "tags"]
        }
        
        response = await es_client.search(index="kb", body=search_body)
        
        documents = []
        for hit in response['hits']['hits']:
            content = hit['_source'].get('content', '')
            if content:
                documents.append(content)
        
        return documents
    
    except Exception as e:
        print(f"Search error: {e}")
        return []

async def get_document_count() -> int:
    """Get total number of documents in knowledge base"""
    if not es_client:
        return 0
    
    try:
        response = await es_client.count(index="kb")
        return response['count']
    except Exception as e:
        print(f"Count error: {e}")
        return 0