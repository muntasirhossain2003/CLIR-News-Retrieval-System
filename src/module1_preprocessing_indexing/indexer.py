"""
Indexer for building Whoosh (keyword-based) and FAISS (embedding-based) indices
"""

import os
import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict

import faiss
from whoosh import index
from whoosh.fields import Schema, ID, TEXT, STORED
from whoosh.analysis import RegexTokenizer, LowercaseFilter
from whoosh.analysis.analyzers import CompositeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class Indexer:
    """
    Build search indices for cross-lingual information retrieval.
    Creates both Whoosh (BM25) and FAISS (dense vector) indices.
    """
    
    def __init__(self):
        self.articles = []
        self.whoosh_index = None
        self.faiss_index = None
    
    def load_data(self, pickle_file: str) -> List[Dict]:
        """
        Load processed articles from pickle file.
        
        Args:
            pickle_file: Path to pickle file with articles and embeddings
            
        Returns:
            List of article dictionaries
        """
        logging.info(f"Loading data from {pickle_file}")
        
        if not os.path.exists(pickle_file):
            raise FileNotFoundError(f"Pickle file not found: {pickle_file}")
        
        with open(pickle_file, 'rb') as f:
            self.articles = pickle.load(f)
        
        logging.info(f"Loaded {len(self.articles)} articles")
        
        # Validate data
        if not self.articles:
            raise ValueError("No articles found in pickle file")
        
        # Check required fields
        required_fields = ['url', 'title', 'body', 'embedding', 'filepath']
        missing_fields = []
        
        for field in required_fields:
            if field not in self.articles[0]:
                missing_fields.append(field)
        
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        return self.articles
    
    def build_whoosh_index(self, index_dir: str = "data/indices/whoosh"):
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        multilingual_analyzer = CompositeAnalyzer(
            RegexTokenizer(r'\w+'),
            LowercaseFilter()
        )
        
        schema = Schema(
            url=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=multilingual_analyzer),
            body=TEXT(stored=True, analyzer=multilingual_analyzer),
            path=STORED()
        )
        
        if index.exists_in(str(index_path)):
            self.whoosh_index = index.open_dir(str(index_path))
        else:
            self.whoosh_index = index.create_in(str(index_path), schema)
        
        writer = self.whoosh_index.writer()
        
        for i, article in enumerate(self.articles):
            try:
                writer.add_document(
                    url=article['url'],
                    title=article.get('title', ''),
                    body=article.get('body', ''),
                    path=article.get('filepath', '')
                )
            except Exception as e:
                logging.error(f"Error indexing article: {e}")
        
        writer.commit()
        return self.whoosh_index
    
    def build_faiss_index(self, index_file: str = "data/indices/faiss_index.bin"):
        index_path = Path(index_file)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        embeddings = []
        for article in self.articles:
            embedding = article.get('embedding')
            if embedding is None:
                raise ValueError(f"Article missing embedding: {article.get('url', 'unknown')}")
            embeddings.append(embedding)
        
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_matrix)
        
        dimension = embeddings_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings_matrix)
        
        faiss.write_index(self.faiss_index, str(index_file))
        return self.faiss_index


def build_indices(
    pickle_file: str = "data/embeddings/articles_with_embeddings.pkl",
    whoosh_dir: str = "data/indices/whoosh",
    faiss_file: str = "data/indices/faiss_index.bin"
):
    indexer = Indexer()
    articles = indexer.load_data(pickle_file)
    whoosh_index = indexer.build_whoosh_index(whoosh_dir)
    faiss_index = indexer.build_faiss_index(faiss_file)
    
    return whoosh_index, faiss_index


if __name__ == "__main__":
    build_indices(
        pickle_file="data/embeddings/articles_with_embeddings.pkl",
        whoosh_dir="data/indices/whoosh",
        faiss_file="data/indices/faiss_index.bin"
    )
