import numpy as np
from typing import List, Dict, Set


class Ranker:
    """Handles score normalization, fusion, and evaluation metrics."""
    
    def __init__(self):
        pass
    
    def normalize_scores(self, results: List[Dict]) -> List[Dict]:
        """Apply Min-Max normalization to scale scores to [0, 1]."""
            return results
        
        # Extract scores
        scores = [r['score'] for r in results]
        
        if not scores:
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            for result in results:
                result['normalized_score'] = 1.0 if max_score > 0 else 0.0
        else:
            for result in results:
                normalized = (result['score'] - min_score) / (max_score - min_score)
                result['normalized_score'] = normalized
        
        return results
    
    def merge_and_rank(
        self, 
        lexical_results: List[Dict], 
        semantic_results: List[Dict], 
        alpha: float = 0.5
    ) -> Dict:
        """Merge lexical and semantic results using weighted fusion."""
        lexical_normalized = self.normalize_scores(lexical_results.copy())
        semantic_normalized = self.normalize_scores(semantic_results.copy())
        
        # Create dictionaries for quick lookup
        lexical_dict = {r['url']: r['normalized_score'] for r in lexical_normalized}
        semantic_dict = {r['url']: r['normalized_score'] for r in semantic_normalized}
        
        all_urls = set(lexical_dict.keys()) | set(semantic_dict.keys())
        merged = []
        for url in all_urls:
            lexical_score = lexical_dict.get(url, 0.0)
            semantic_score = semantic_dict.get(url, 0.0)
            
            final_score = (alpha * semantic_score) + ((1 - alpha) * lexical_score)
            article = None
            for r in semantic_normalized:
                if r['url'] == url:
                    article = r
                    break
            if article is None:
                for r in lexical_normalized:
                    if r['url'] == url:
                        article = r
                        break
            
            if article:
                merged.append({
                    'url': url,
                    'title': article.get('title', ''),
                    'lang': article.get('lang', ''),
                    'lexical_score': lexical_score,
                    'semantic_score': semantic_score,
                    'final_score': final_score
                })
        
        merged.sort(key=lambda x: x['final_score'], reverse=True)
        warning = None
        if merged and merged[0]['final_score'] < 0.2:
            warning = "Low confidence: Top result has score < 0.2"
        
        return {
            'results': merged,
            'warning': warning
        }
    
    def calculate_metrics(
        self, 
        retrieved_docs: List[str], 
        relevant_docs_ids: Set[str]
    ) -> Dict[str, float]:
        """Calculate evaluation metrics: Precision@10, Recall@50, MRR, nDCG@10."""
        top_10 = retrieved_docs[:10]
        relevant_in_top_10 = sum(1 for doc in top_10 if doc in relevant_docs_ids)
        metrics['precision@10'] = relevant_in_top_10 / 10.0 if len(top_10) >= 10 else relevant_in_top_10 / len(top_10)
        top_50 = retrieved_docs[:50]
        relevant_in_top_50 = sum(1 for doc in top_50 if doc in relevant_docs_ids)
        total_relevant = len(relevant_docs_ids)
        metrics['recall@50'] = relevant_in_top_50 / total_relevant if total_relevant > 0 else 0.0
        mrr = 0.0
        for rank, doc in enumerate(retrieved_docs, start=1):
            if doc in relevant_docs_ids:
                mrr = 1.0 / rank
                break
        metrics['mrr'] = mrr
        metrics['ndcg@10'] = self._calculate_ndcg(retrieved_docs[:10], relevant_docs_ids, k=10)
        
        return metrics
    
    def _calculate_ndcg(self, retrieved_docs: List[str], relevant_docs_ids: Set[str], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        dcg = 0.0
        for i, doc in enumerate(retrieved_docs[:k], start=1):
            relevance = 1 if doc in relevant_docs_ids else 0
            dcg += relevance / np.log2(i + 1)
        
        # IDCG (Ideal DCG) - assuming binary relevance
        num_relevant = min(len(relevant_docs_ids), k)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, num_relevant + 1))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
