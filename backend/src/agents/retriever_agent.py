from sentence_transformers import CrossEncoder
from .state import LOOKUPState
from src.kg_holder import get_kg

# Load cross-encoder reranker once (global to avoid reloading)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def retriever_agent(state: LOOKUPState) -> dict:
    """
    Retriever agent:
    - Uses the vector store to find news articles relevant to the query.
    - Reranks the top results using a cross-encoder.
    """
    kg = get_kg()
    if kg is None or not hasattr(kg, 'vector_store'):
        return {"reranked_documents": []}

    query = state["query"]
    # Initial retrieval (e.g., top 20)
    initial_results = kg.vector_store.search(query, n_results=20)
    if not initial_results:
        return {"reranked_documents": []}

    # Prepare pairs for cross-encoder: (query, document)
    pairs = [(query, res['document']) for res in initial_results]
    scores = reranker.predict(pairs)

    # Combine scores with results and sort
    for res, score in zip(initial_results, scores):
        res['rerank_score'] = float(score)

    reranked = sorted(initial_results, key=lambda x: x['rerank_score'], reverse=True)

    # Keep top 5 after reranking
    top_docs = reranked[:5]

    # Format for state (simple list of strings)
    formatted_docs = [f"{d['document']} (relevance: {d['rerank_score']:.3f})" for d in top_docs]

    return {"reranked_documents": formatted_docs}