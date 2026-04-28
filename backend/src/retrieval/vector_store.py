import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any

class VectorStore:
    """
    Wrapper for ChromaDB to store and retrieve financial news.
    Uses sentence-transformers for embedding documents.
    """
    def __init__(self, collection_name="financial_news", persist_directory="./chroma_db"):
        # The new way to initialize persistent storage
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Using Chroma's default embedding function (sentence-transformers)
        # This is more stable than manually wrapping SentenceTransformer
        self.emb_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.emb_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def clear_all(self):
        self.collection.delete(where={})
    
    def add_document(self, headline: str, metadata: Dict[str, Any]):
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, headline + str(metadata.get('ticker', ''))))
        self.collection.upsert(
            ids=[doc_id],
            metadatas=[metadata],
            documents=[headline]
        )
    
    def search(self, query: str, n_results: int = 10) -> List[Dict]:
        """
        Search for documents similar to the query.
        Returns list of dicts with 'id', 'document', 'metadata', 'distance'.
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["metadatas", "documents", "distances"]
        )
        # Format results
        formatted = []
        for i in range(len(results['ids'][0])):
            formatted.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        return formatted