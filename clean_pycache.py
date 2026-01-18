#!/usr/bin/env python
"""
Script to delete all __pycache__ directories in the repository.
"""

import shutil
from pathlib import Path


def clean_pycache(root_dir: Path = None) -> int:
    """
    Remove all __pycache__ directories recursively.

    Args:
        root_dir: Root directory to start searching. Defaults to script location.

    Returns:
        Number of __pycache__ directories deleted.
    """
    if root_dir is None:
        root_dir = Path(__file__).parent

    count = 0
    for pycache_dir in root_dir.rglob("__pycache__"):
        if pycache_dir.is_dir():
            print(f"Removing: {pycache_dir}")
            shutil.rmtree(pycache_dir)
            count += 1

    return count


if __name__ == "__main__":
    deleted = clean_pycache()
