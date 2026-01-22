import os
import pickle
import numpy as np
import faiss
from typing import List, Dict

from whoosh import index
from whoosh.qparser import MultifieldParser, OrGroup, QueryParser
from sentence_transformers import SentenceTransformer

from src.module2_query_processing.query_processor import QueryProcessor


class Retriever:
    """Retriever for lexical and semantic search."""
    
    def __init__(
        self,
        data_path: str = "data/embeddings/articles_with_embeddings.pkl",
        whoosh_path: str = "data/indices/whoosh",
        faiss_path: str = "data/indices/faiss_index.bin",
        model_name: str = "sentence-transformers/LaBSE"
    ):
        self.query_processor = QueryProcessor()
        self.model = SentenceTransformer(model_name)
        
        # Load Whoosh index
        if not os.path.exists(whoosh_path):
            raise FileNotFoundError(f"Whoosh index not found: {whoosh_path}")
        self.whoosh_index = index.open_dir(whoosh_path)
        
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
        self.faiss_index = faiss.read_index(faiss_path)
        
        # Load metadata
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
        with open(data_path, 'rb') as f:
            self.articles = pickle.load(f)
    
    def search(self, query_text: str, k: int = 10) -> Dict[str, List[Dict]]:
        """Search using both lexical (Whoosh) and semantic (FAISS) methods."""
        query_result = self.query_processor.process_query(query_text)
        original_text = query_result['original_text']
        translated_text = query_result['translated_text']
        
        lexical_results = self._whoosh_search(original_text, translated_text, k)
        semantic_results = self._faiss_search(original_text, k)
        
        return {
            'lexical': lexical_results,
            'semantic': semantic_results
        }
    
    def _whoosh_search(self, original_text: str, translated_text: str, k: int) -> List[Dict]:
        results = []
        all_results = {}
        
        with self.whoosh_index.searcher() as searcher:
            # Phrase search (exact match)
            phrase_queries = []
            if original_text:
                phrase_queries.append(f'title:"{original_text}"^10.0 OR body:"{original_text}"^2.0')
            if translated_text:
                phrase_queries.append(f'title:"{translated_text}"^10.0 OR body:"{translated_text}"^2.0')
            
            if phrase_queries:
                from whoosh.qparser import QueryParser
                phrase_query_string = " OR ".join(phrase_queries)
                try:
                    phrase_parser = QueryParser("body", schema=self.whoosh_index.schema)
                    phrase_query = phrase_parser.parse(phrase_query_string)
                    phrase_results = searcher.search(phrase_query, limit=k*2)
                    
                    for hit in phrase_results:
                        url = hit.get('url', '')
                        if url not in all_results:
                            all_results[url] = {
                                'title': hit.get('title', ''),
                                'url': url,
                                'score': hit.score * 2.0,  # Boost phrase matches
                                'path': hit.get('path', '')
                            }
                except:
                    pass
            
            # OR keyword search
            if original_text or translated_text:
                if translated_text:
                    keyword_query_string = f"({original_text}) OR ({translated_text})"
                else:
                    keyword_query_string = original_text
                
                parser = MultifieldParser(
                    ['title', 'body'], 
                    schema=self.whoosh_index.schema, 
                    group=OrGroup,
                    fieldboosts={'title': 5.0, 'body': 1.0}
                )
                keyword_query = parser.parse(keyword_query_string)
                keyword_results = searcher.search(keyword_query, limit=k*2)
                
                for hit in keyword_results:
                    url = hit.get('url', '')
                    if url not in all_results:
                        all_results[url] = {
                            'title': hit.get('title', ''),
                            'url': url,
                            'score': hit.score,
                            'path': hit.get('path', '')
                        }
                    else:
                        all_results[url]['score'] += hit.score
            
            results = list(all_results.values())
            results.sort(key=lambda x: x['score'], reverse=True)
            search_results = results[:k]
            final_results = []
            for result in search_results:
                final_results.append({
                    'title': result['title'],
                    'url': result['url'],
                    'score': result['score'],
                    'lang': self._extract_lang_from_path(result['path'])
                })
        
        return final_results
    
    def _faiss_search(self, query_text: str, k: int) -> List[Dict]:
        query_embedding = self.model.encode([query_text], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.articles):
                article = self.articles[idx]
                results.append({
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'score': float(score),
                    'lang': article.get('language', '')
                })
        
        return results
    
    def _extract_lang_from_path(self, path: str) -> str:
        """Extract language from file path."""
        if 'bangla' in path.lower():
            return 'bangla'
        elif 'english' in path.lower():
            return 'english'
        return ''
