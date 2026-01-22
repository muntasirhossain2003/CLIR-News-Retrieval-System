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
from whoosh.analysis import RegexTokenizer, LowercaseFilter, StopFilter
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
        """
        Build Whoosh index for keyword-based search (BM25).
        
        Args:
            index_dir: Directory to save Whoosh index
        """
        logging.info("Building Whoosh index...")
        
        # Create index directory
        index_path = Path(index_dir)
        index_path.mkdir(parents=True, exist_ok=True)
        
        # Define multilingual analyzer for Bangla and English
        # Use regex tokenizer that works with Unicode characters
        multilingual_analyzer = CompositeAnalyzer(
            RegexTokenizer(r'\w+'),  # Matches Unicode word characters including Bangla
            LowercaseFilter()
        )
        
        # Define schema
        schema = Schema(
            url=ID(stored=True, unique=True),
            title=TEXT(stored=True, analyzer=multilingual_analyzer),
            body=TEXT(stored=True, analyzer=multilingual_analyzer),
            path=STORED()  # Map filepath to path field
        )
        
        # Create index
        if index.exists_in(str(index_path)):
            logging.warning(f"Index already exists at {index_dir}, overwriting...")
            self.whoosh_index = index.open_dir(str(index_path))
        else:
            self.whoosh_index = index.create_in(str(index_path), schema)
        
        # Index documents
        writer = self.whoosh_index.writer()
        
        for i, article in enumerate(self.articles):
            try:
                writer.add_document(
                    url=article['url'],
                    title=article.get('title', ''),
                    body=article.get('body', ''),
                    path=article.get('filepath', '')  # Map filepath -> path
                )
                
                if (i + 1) % 500 == 0:
                    logging.info(f"Indexed {i + 1}/{len(self.articles)} documents")
                    
            except Exception as e:
                logging.error(f"Error indexing article {article.get('url', 'unknown')}: {e}")
        
        writer.commit()
        logging.info(f"Whoosh index built successfully: {len(self.articles)} documents")
        logging.info(f"Index saved to: {index_dir}")
        
        return self.whoosh_index
    
    def build_faiss_index(self, index_file: str = "data/indices/faiss_index.bin"):
        """
        Build FAISS index for dense vector similarity search.
        Uses IndexFlatIP (Inner Product) with L2-normalized vectors
        so that Inner Product equals Cosine Similarity.
        
        Args:
            index_file: Path to save FAISS index file
        """
        logging.info("Building FAISS index...")
        
        # Create output directory
        index_path = Path(index_file)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract embeddings
        embeddings = []
        for article in self.articles:
            embedding = article.get('embedding')
            if embedding is None:
                raise ValueError(f"Article missing embedding: {article.get('url', 'unknown')}")
            embeddings.append(embedding)
        
        # Convert to numpy array (float32 for FAISS)
        embeddings_matrix = np.array(embeddings, dtype=np.float32)
        
        logging.info(f"Embedding matrix shape: {embeddings_matrix.shape}")
        
        # L2 normalization
        # After normalization: ||x|| = 1, so x·y = cos(θ)
        logging.info("Normalizing embeddings (L2)...")
        faiss.normalize_L2(embeddings_matrix)
        
        # Create FAISS index (Inner Product)
        dimension = embeddings_matrix.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Add vectors to index
        self.faiss_index.add(embeddings_matrix)
        
        logging.info(f"FAISS index built: {self.faiss_index.ntotal} vectors of dimension {dimension}")
        
        # Save index
        faiss.write_index(self.faiss_index, str(index_file))
        logging.info(f"FAISS index saved to: {index_file}")
        
        return self.faiss_index


def build_indices(
    pickle_file: str = "data/embeddings/articles_with_embeddings.pkl",
    whoosh_dir: str = "data/indices/whoosh",
    faiss_file: str = "data/indices/faiss_index.bin"
):
    """
    Main function to build both Whoosh and FAISS indices.
    
    Args:
        pickle_file: Path to processed articles pickle file
        whoosh_dir: Directory for Whoosh index
        faiss_file: Path for FAISS index file
    """
    logging.info("="*60)
    logging.info("BUILDING SEARCH INDICES")
    logging.info("="*60)
    
    # Initialize indexer
    indexer = Indexer()
    
    # Load data
    articles = indexer.load_data(pickle_file)
    
    # Build Whoosh index
    logging.info("\n" + "-"*60)
    logging.info("STEP 1: Building Whoosh Index (BM25)")
    logging.info("-"*60)
    whoosh_index = indexer.build_whoosh_index(whoosh_dir)
    
    # Build FAISS index
    logging.info("\n" + "-"*60)
    logging.info("STEP 2: Building FAISS Index (Dense Vectors)")
    logging.info("-"*60)
    faiss_index = indexer.build_faiss_index(faiss_file)
    
    # Summary
    logging.info("\n" + "="*60)
    logging.info("INDEX BUILDING COMPLETE")
    logging.info("="*60)
    logging.info(f"Total articles indexed: {len(articles)}")
    logging.info(f"Whoosh index: {whoosh_dir}")
    logging.info(f"FAISS index: {faiss_file}")
    logging.info(f"FAISS vectors: {faiss_index.ntotal}")
    logging.info(f"Vector dimension: {faiss_index.d}")
    logging.info("="*60)
    
    return whoosh_index, faiss_index


if __name__ == "__main__":
    # Build indices
    build_indices(
        pickle_file="data/embeddings/articles_with_embeddings.pkl",
        whoosh_dir="data/indices/whoosh",
        faiss_file="data/indices/faiss_index.bin"
    )
