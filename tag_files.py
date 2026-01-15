"""
Script to automatically tag files in the 'data' folder with random Access Control Levels.
Use this to prepare a test dataset for the Access Control RAG system.

Levels:
- [L1]: Public
- [L2]: Internal (Default)
- [L3]: Confidential
- [L4]: Sensitive
- [L5]: Top Secret
"""

import os
import random
import shutil
from pathlib import Path

# Configuration
DATA_DIR = Path("data")
LEVELS = [
    "[L1]",  # Public
    "[L2]",  # Internal
    "[L3]",  # Confidential
    "[L4]",  # Sensitive
    "[L5]",  # Top Secret
]
# Probabilities for each level (L2 is most common)
PROBABILITIES = [0.1, 0.4, 0.2, 0.2, 0.1]

def tag_files():
    if not DATA_DIR.exists():
        print(f"❌ '{DATA_DIR}' directory not found.")
        return

    files = [f for f in DATA_DIR.iterdir() if f.is_file()]
    if not files:
        print("❌ No files found in data directory.")
        return

    print(f"Found {len(files)} files to process...\n")
    
    renamed_count = 0
    skipped_count = 0

    for file_path in files:
        # Check if file is already tagged
        if any(tag in file_path.name for tag in LEVELS):
            print(f"⏩ Skipping {file_path.name} (already tagged)")
            skipped_count += 1
            continue

        # Choose a random level
        level_tag = random.choices(LEVELS, weights=PROBABILITIES, k=1)[0]
        
        # Construct new name: 'report.pdf' -> 'report_[L3].pdf'
        stem = file_path.stem
        suffix = file_path.suffix
        new_name = f"{stem}_{level_tag}{suffix}"
        new_path = file_path.parent / new_name
        
        try:
            file_path.rename(new_path)
            print(f"✓ Renamed: {file_path.name} -> {new_name}")
            renamed_count += 1
        except Exception as e:
            print(f"❌ Error renaming {file_path.name}: {e}")

    print("\n" + "="*40)
    print(f"Summary:")
    print(f"  Tagged:  {renamed_count}")
    print(f"  Skipped: {skipped_count}")
    print("="*40)

if __name__ == "__main__":
    tag_files()
