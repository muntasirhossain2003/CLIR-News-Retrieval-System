"""
Unified Indexing Launcher

Launches the indexing pipeline from Module A.
This is the main entry point for building WHOOSH and FAISS indexes.
"""

import os
import sys
from pathlib import Path

# Calculate project root (4 levels up from this file)
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent.parent
INDEXING_DIR = CURRENT_FILE.parent

# Add paths
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INDEXING_DIR))

# Change to project root so relative paths work
os.chdir(PROJECT_ROOT)

if __name__ == "__main__":
    # Import and run build_index
    try:
        from build_index import main as build_main

        print("=" * 70)
        print("CLIR Unified Indexing System")
        print("=" * 70)
        print("")
        print("Architecture:")
        print("  Module A: Indexing (WHOOSH lexical + FAISS semantic)")
        print("  Module B: Query preprocessing")
        print("  Module C: Retrieval (uses Module A's indexes)")
        print("")
        print("=" * 70)
        print("")

        # Run Module A's indexing
        build_main()

    except ImportError as e:
        print(f"ERROR: Failed to load build_index: {e}")
        print("")
        print("Run from Module A indexing directory:")
        print(f"  cd '{INDEXING_DIR}'")
        print(f"  python build_unified_indexes.py --data data/metadata.csv")
        sys.exit(1)
