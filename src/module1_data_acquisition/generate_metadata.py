import os
import json
import pandas as pd
from glob import glob
from tqdm import tqdm

def generate_metadata():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    data_root = os.path.join(project_root, "data", "raw")
    all_files = glob(os.path.join(data_root, "*", "*", "*.json"))
    
    metadata = []
    
    print(f"Found {len(all_files)} files. Generating metadata...")
    
    if len(all_files) == 0:
        print("No JSON files found in data/raw. Check your cwd or crawl data first.")
        return

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
    
    # Cleaning steps
    print(f"Initial count: {len(df)}")
    
    if df.empty:
        print("DataFrame is empty.")
        return

    # Drop duplicates by URL
    if 'url' in df.columns:
        df = df.drop_duplicates(subset=['url'])
        print(f"After removing duplicates: {len(df)}")
        
    # Drop entries with missing titles or URLs
    if 'url' in df.columns and 'title' in df.columns:
        df = df.dropna(subset=['url', 'title'])
        print(f"After removing missing title/url: {len(df)}")
    
    # Sort
    if 'language' in df.columns and 'source' in df.columns:
        df = df.sort_values(by=['language', 'source'], ascending=[True, True])

    output_path = os.path.join(project_root, "data", "metadata.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved cleaned metadata to {output_path}")
    print(df.groupby(['language', 'source']).size())

if __name__ == "__main__":
    generate_metadata()
