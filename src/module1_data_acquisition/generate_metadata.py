import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm

def generate_metadata():
    data_root = os.path.join("data", "raw")
    all_files = glob(os.path.join(data_root, "*", "*", "*.json"))
    
    metadata = []
    
    print(f"Found {len(all_files)} files. Generating metadata...")
    
    for filepath in tqdm(all_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            entry = {
                'filename': os.path.basename(filepath),
                'language': data.get('language'),
                'source': data.get('source'),
                'url': data.get('url'),
                'title': data.get('title'),
                'date': data.get('date'),
                'crawled_at': data.get('crawled_at'),
                'filepath': filepath
            }
            metadata.append(entry)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    df = pd.DataFrame(metadata)
    output_path = os.path.join("data", "metadata.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved metadata to {output_path}")
    print(df.groupby(['language', 'source']).size())

if __name__ == "__main__":
    generate_metadata()
