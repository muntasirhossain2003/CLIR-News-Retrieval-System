"""
Embedding generation using LaBSE model for cross-lingual retrieval
"""

import pickle
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .utils import load_all_articles, count_tokens, clean_text, filter_articles, deduplicate_articles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class EmbeddingGenerator:
    """
    Generate embeddings for articles using LaBSE model.
    LaBSE (Language-agnostic BERT Sentence Embedding) is specifically designed
    for cross-lingual retrieval tasks.
    """
    
    def __init__(self, model_name: str = 'sentence-transformers/LaBSE'):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
        """
        logging.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logging.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def process_articles(
        self, 
        data_dir: str = "data/raw",
        min_tokens: int = 50,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Load, process, and generate embeddings for all articles.
        
        Args:
            data_dir: Directory containing raw JSON files
            min_tokens: Minimum token count for filtering
            batch_size: Batch size for embedding generation
            
        Returns:
            List of processed articles with embeddings
        """
        # Load all articles
        logging.info("Loading articles...")
        articles = load_all_articles(data_dir)
        
        if not articles:
            logging.error("No articles found!")
            return []
        
        # Deduplicate
        logging.info("Deduplicating articles...")
        articles = deduplicate_articles(articles)
        
        # Add token count
        logging.info("Adding token counts...")
        for article in tqdm(articles, desc="Counting tokens"):
            body = article.get('body', '')
            cleaned_body = clean_text(body)
            article['body'] = cleaned_body  # Update with cleaned version
            article['token_count'] = count_tokens(cleaned_body)
            
            # Also clean title
            if 'title' in article:
                article['title'] = clean_text(article['title'])
        
        # Filter articles
        logging.info(f"Filtering articles (min tokens: {min_tokens})...")
        articles = filter_articles(articles, min_tokens=min_tokens)
        
        # Generate embeddings
        logging.info("Generating embeddings with LaBSE...")
        articles = self.generate_embeddings(articles, batch_size=batch_size)
        
        return articles
    
    def generate_embeddings(self, articles: List[Dict], batch_size: int = 32) -> List[Dict]:
        """
        Generate embeddings for article bodies.
        
        Args:
            articles: List of article dictionaries
            batch_size: Batch size for encoding
            
        Returns:
            Articles with added 'embedding' field
        """
        # Extract texts for embedding (use body for now)
        texts = [article['body'] for article in articles]
        
        logging.info(f"Generating embeddings for {len(texts)} articles...")
        
        # Generate embeddings in batches with progress bar
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Add embeddings to articles
        for article, embedding in zip(articles, embeddings):
            article['embedding'] = embedding
        
        logging.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        return articles


def generate_embeddings(
    data_dir: str = "data/raw",
    output_file: str = "data/embeddings/articles_with_embeddings.pkl",
    model_name: str = 'sentence-transformers/LaBSE',
    min_tokens: int = 50,
    batch_size: int = 32
) -> str:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generator = EmbeddingGenerator(model_name=model_name)
    
    articles = generator.process_articles(
        data_dir=data_dir,
        min_tokens=min_tokens,
        batch_size=batch_size
    )
    
    if not articles:
        return ""
    
    save_embeddings(articles, output_file)
    return output_file


def save_embeddings(articles: List[Dict], output_file: str):
    with open(output_file, 'wb') as f:
        pickle.dump(articles, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_embeddings(pickle_file: str) -> List[Dict]:
    with open(pickle_file, 'rb') as f:
        articles = pickle.load(f)
    return articles


if __name__ == "__main__":
    generate_embeddings(
        data_dir="data/raw",
        output_file="data/embeddings/articles_with_embeddings.pkl",
        model_name='sentence-transformers/LaBSE',
        min_tokens=50,
        batch_size=32
    )
