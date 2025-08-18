#!/usr/bin/env python3
"""
Advanced Retrieval Module
Implements hybrid search (dense + sparse) for improved query understanding and context retrieval
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    retrieval_type: str  # "dense", "sparse", or "hybrid"


class AdvancedRetrieval:
    """
    Advanced retrieval system with hybrid search capabilities
    Combines dense (semantic) and sparse (keyword-based) search for optimal results
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self._fitted = False
        self._documents = []
        self._embeddings = []
        self._tfidf_matrix = None
        
    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the retrieval system"""
        try:
            self._documents = documents
            
            # Generate dense embeddings
            texts = [doc.get("content", "") for doc in documents]
            self._embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Generate sparse TF-IDF representations
            self._tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            self._fitted = True
            
            logger.info(f"Added {len(documents)} documents to advanced retrieval system")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
            
    def hybrid_search(self, query: str, top_k: int = 5, 
                     dense_weight: float = 0.7, sparse_weight: float = 0.3) -> List[RetrievalResult]:
        """
        Perform hybrid search combining dense and sparse retrieval
        """
        try:
            if not self._fitted:
                raise ValueError("Retrieval system not initialized. Call add_documents() first.")
                
            # Dense search (semantic)
            dense_results = self._dense_search(query, top_k * 2)
            
            # Sparse search (keyword-based)
            sparse_results = self._sparse_search(query, top_k * 2)
            
            # Combine results using weighted scoring
            combined_results = self._combine_results(
                dense_results, sparse_results, dense_weight, sparse_weight
            )
            
            # Return top-k results
            return combined_results[:top_k]
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
            
    def _dense_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform dense (semantic) search"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_embedding, self._embeddings)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append(RetrievalResult(
                        content=self._documents[idx].get("content", ""),
                        score=float(similarities[idx]),
                        source=self._documents[idx].get("source", "unknown"),
                        metadata=self._documents[idx].get("metadata", {}),
                        retrieval_type="dense"
                    ))
                    
            return results
            
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
            
    def _sparse_search(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Perform sparse (keyword-based) search"""
        try:
            # Transform query using fitted TF-IDF vectorizer
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self._tfidf_matrix)[0]
            
            # Get top-k results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    results.append(RetrievalResult(
                        content=self._documents[idx].get("content", ""),
                        score=float(similarities[idx]),
                        source=self._documents[idx].get("source", "unknown"),
                        metadata=self._documents[idx].get("metadata", {}),
                        retrieval_type="sparse"
                    ))
                    
            return results
            
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
            
    def _combine_results(self, dense_results: List[RetrievalResult], 
                        sparse_results: List[RetrievalResult],
                        dense_weight: float, sparse_weight: float) -> List[RetrievalResult]:
        """Combine dense and sparse results using weighted scoring"""
        try:
            # Create a combined results dictionary
            combined_dict = {}
            
            # Add dense results
            for result in dense_results:
                key = result.content[:100]  # Use first 100 chars as key
                combined_dict[key] = {
                    "content": result.content,
                    "dense_score": result.score,
                    "sparse_score": 0.0,
                    "source": result.source,
                    "metadata": result.metadata,
                    "retrieval_type": "hybrid"
                }
                
            # Add sparse results
            for result in sparse_results:
                key = result.content[:100]  # Use first 100 chars as key
                if key in combined_dict:
                    combined_dict[key]["sparse_score"] = result.score
                else:
                    combined_dict[key] = {
                        "content": result.content,
                        "dense_score": 0.0,
                        "sparse_score": result.score,
                        "source": result.source,
                        "metadata": result.metadata,
                        "retrieval_type": "hybrid"
                    }
                    
            # Calculate combined scores
            combined_results = []
            for key, data in combined_dict.items():
                combined_score = (
                    data["dense_score"] * dense_weight + 
                    data["sparse_score"] * sparse_weight
                )
                
                combined_results.append(RetrievalResult(
                    content=data["content"],
                    score=combined_score,
                    source=data["source"],
                    metadata=data["metadata"],
                    retrieval_type="hybrid"
                ))
                
            # Sort by combined score
            combined_results.sort(key=lambda x: x.score, reverse=True)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Result combination failed: {e}")
            return []
            
    def semantic_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Perform semantic search only"""
        return self._dense_search(query, top_k)
        
    def keyword_search(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Perform keyword search only"""
        return self._sparse_search(query, top_k)
        
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get retrieval system statistics"""
        return {
            "total_documents": len(self._documents),
            "embedding_dimensions": self._embeddings.shape[1] if len(self._embeddings) > 0 else 0,
            "tfidf_features": self._tfidf_matrix.shape[1] if self._tfidf_matrix is not None else 0,
            "fitted": self._fitted
        }


class ContextualRetrieval:
    """
    Contextual retrieval system that considers query context and user preferences
    """
    
    def __init__(self, advanced_retrieval: AdvancedRetrieval):
        self.advanced_retrieval = advanced_retrieval
        self.context_cache = {}
        
    async def retrieve_with_context(self, query: str, context: Dict[str, Any], 
                                  top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve results considering query context and user preferences
        """
        try:
            # Extract context information
            user_preferences = context.get("user_preferences", {})
            conversation_history = context.get("conversation_history", [])
            domain_expertise = context.get("domain_expertise", {})
            
            # Enhance query with context
            enhanced_query = self._enhance_query_with_context(
                query, user_preferences, conversation_history, domain_expertise
            )
            
            # Perform hybrid search
            results = self.advanced_retrieval.hybrid_search(enhanced_query, top_k)
            
            # Re-rank results based on context
            reranked_results = self._rerank_with_context(
                results, context
            )
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Contextual retrieval failed: {e}")
            return []
            
    def _enhance_query_with_context(self, query: str, user_preferences: Dict,
                                  conversation_history: List, domain_expertise: Dict) -> str:
        """Enhance query with contextual information"""
        enhanced_parts = [query]
        
        # Add user preferences
        if user_preferences.get("preferred_complexity"):
            enhanced_parts.append(f"complexity: {user_preferences['preferred_complexity']}")
            
        if user_preferences.get("preferred_visualization"):
            enhanced_parts.append(f"visualization: {user_preferences['preferred_visualization']}")
            
        # Add recent conversation context
        if conversation_history:
            recent_context = " ".join([msg.get("content", "") for msg in conversation_history[-3:]])
            enhanced_parts.append(f"context: {recent_context}")
            
        # Add domain expertise
        if domain_expertise.get("expertise_level"):
            enhanced_parts.append(f"expertise: {domain_expertise['expertise_level']}")
            
        return " ".join(enhanced_parts)
        
    def _rerank_with_context(self, results: List[RetrievalResult], 
                           context: Dict[str, Any]) -> List[RetrievalResult]:
        """Re-rank results based on context"""
        try:
            user_preferences = context.get("user_preferences", {})
            
            for result in results:
                # Adjust score based on user preferences
                if user_preferences.get("preferred_complexity"):
                    complexity_match = self._check_complexity_match(
                        result.content, user_preferences["preferred_complexity"]
                    )
                    result.score *= (1.0 + complexity_match * 0.2)
                    
                if user_preferences.get("preferred_visualization"):
                    viz_match = self._check_visualization_match(
                        result.content, user_preferences["preferred_visualization"]
                    )
                    result.score *= (1.0 + viz_match * 0.1)
                    
            # Re-sort by adjusted scores
            results.sort(key=lambda x: x.score, reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Context re-ranking failed: {e}")
            return results
            
    def _check_complexity_match(self, content: str, preferred_complexity: str) -> float:
        """Check if content matches preferred complexity level"""
        complexity_keywords = {
            "simple": ["basic", "simple", "overview", "summary"],
            "medium": ["detailed", "analysis", "comparison", "trend"],
            "complex": ["advanced", "complex", "sophisticated", "detailed analysis"]
        }
        
        content_lower = content.lower()
        keywords = complexity_keywords.get(preferred_complexity.lower(), [])
        
        match_count = sum(1 for keyword in keywords if keyword in content_lower)
        return min(match_count / len(keywords), 1.0) if keywords else 0.0
        
    def _check_visualization_match(self, content: str, preferred_visualization: str) -> float:
        """Check if content matches preferred visualization type"""
        viz_keywords = {
            "chart": ["chart", "graph", "plot", "visualization"],
            "table": ["table", "grid", "tabular", "data table"],
            "dashboard": ["dashboard", "overview", "summary", "metrics"]
        }
        
        content_lower = content.lower()
        keywords = viz_keywords.get(preferred_visualization.lower(), [])
        
        match_count = sum(1 for keyword in keywords if keyword in content_lower)
        return min(match_count / len(keywords), 1.0) if keywords else 0.0
