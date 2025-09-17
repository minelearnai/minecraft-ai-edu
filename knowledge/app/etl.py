import os
from elasticsearch import Elasticsearch
from datetime import datetime
import json

def get_educational_content():
    """Sample educational content for knowledge base"""
    return [
        {
            "id": "math_001",
            "subject": "mathematics",
            "topic": "geometry",
            "title": "Building Squares in Minecraft",
            "content": "To build a square in Minecraft, place blocks in equal rows and columns. A 5x5 square uses 25 blocks total. The perimeter is 4 × side length.",
            "age_group": "10-15",
            "difficulty": "beginner",
            "minecraft_commands": ["/fill ~0 ~0 ~0 ~4 ~0 ~4 stone"],
            "learning_objectives": ["Understanding area", "Calculating perimeter", "Spatial reasoning"]
        },
        {
            "id": "physics_001", 
            "subject": "physics",
            "topic": "electricity",
            "title": "Redstone Circuits Basics",
            "content": "Redstone works like electricity in Minecraft. Power travels through redstone dust up to 15 blocks. Use repeaters to extend signals.",
            "age_group": "12-15",
            "difficulty": "intermediate",
            "minecraft_commands": ["/give @p redstone 64", "/give @p repeater 10"],
            "learning_objectives": ["Basic circuits", "Signal transmission", "Logic gates"]
        },
        {
            "id": "math_002",
            "subject": "mathematics", 
            "topic": "volume",
            "title": "3D Shapes and Volume",
            "content": "Volume measures space inside 3D shapes. A cube with side length 3 has volume 3×3×3 = 27 blocks. Try building different sized cubes!",
            "age_group": "11-15",
            "difficulty": "beginner",
            "minecraft_commands": ["/fill ~0 ~0 ~0 ~2 ~2 ~2 glass"],
            "learning_objectives": ["Volume calculation", "3D visualization", "Mathematical reasoning"]
        }
    ]

def initialize_elasticsearch():
    """Initialize connection to Elasticsearch"""
    es_url = os.getenv('ELASTICSEARCH_URL', 'http://localhost:9200')
    es = Elasticsearch([es_url])
    
    # Test connection
    if not es.ping():
        raise Exception(f"Could not connect to Elasticsearch at {es_url}")
    
    print(f"✅ Connected to Elasticsearch at {es_url}")
    return es

def create_knowledge_index(es):
    """Create the knowledge base index with proper mapping"""
    index_name = 'minecraft_edu_kb'
    
    # Delete index if exists (for development)
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)
        print(f"🗑️  Deleted existing index: {index_name}")
    
    # Create index with mapping
    mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "subject": {"type": "keyword"},
                "topic": {"type": "keyword"}, 
                "title": {"type": "text", "analyzer": "english"},
                "content": {"type": "text", "analyzer": "english"},
                "age_group": {"type": "keyword"},
                "difficulty": {"type": "keyword"},
                "minecraft_commands": {"type": "keyword"},
                "learning_objectives": {"type": "text"},
                "created_at": {"type": "date"}
            }
        }
    }
    
    es.indices.create(index=index_name, body=mapping)
    print(f"📚 Created knowledge base index: {index_name}")
    return index_name

def load_content(es, index_name):
    """Load educational content into Elasticsearch"""
    content_items = get_educational_content()
    
    for item in content_items:
        # Add timestamp
        item['created_at'] = datetime.now().isoformat()
        
        # Index document
        es.index(
            index=index_name,
            id=item['id'],
            body=item
        )
        print(f"📝 Indexed: {item['title']}")
    
    # Refresh index to make documents searchable
    es.indices.refresh(index=index_name)
    print(f"🔄 Refreshed index with {len(content_items)} documents")

def run_etl():
    """Main ETL process"""
    print("🚀 Starting Knowledge Base ETL Process...")
    
    try:
        # Initialize Elasticsearch
        es = initialize_elasticsearch()
        
        # Create index
        index_name = create_knowledge_index(es)
        
        # Load content
        load_content(es, index_name)
        
        # Verify
        count = es.count(index=index_name)['count']
        print(f"✅ ETL Complete! {count} documents in knowledge base")
        
        return "success"
        
    except Exception as e:
        print(f"❌ ETL Failed: {str(e)}")
        return "error"

if __name__ == "__main__":
    result = run_etl()
    exit(0 if result == "success" else 1)