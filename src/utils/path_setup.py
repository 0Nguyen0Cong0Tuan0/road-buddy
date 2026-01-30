"""
Path Setup Utility.

Adds the project root to sys.path for running scripts directly.
This solves the "No module named 'src'" error when running files from subdirectories.

Usage:
    # At the top of any Python file that needs to be run directly:
    if __name__ == "__main__":
        import sys
        from pathlib import Path
        # Add project root to path
        project_root = Path(__file__).resolve().parents[2]  # Adjust number based on depth
        sys.path.insert(0, str(project_root))
"""

import sys
from pathlib import Path
import inspect

def setup_project_path(levels_up: int = 2) -> Path:
    """
    Add project root to sys.path.
    
    Call this from the if __name__ == "__main__" block before any src imports.
    
    Args:
        levels_up: Number of parent directories to go up to find project root.
                   - src/utils/*.py -> 2
                   - src/reasoning/*.py -> 2
                   - src/perception/query_analyzer/*.py -> 3
    
    Returns:
        Path to project root
    """
    # Get the calling frame to determine file location
    caller_frame = inspect.stack()[1]
    caller_file = Path(caller_frame.filename).resolve()
    
    # Go up to project root
    project_root = caller_file
    for _ in range(levels_up):
        project_root = project_root.parent
    
    # Add to sys.path if not already there
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    return project_root

def get_project_root() -> Path:
    """
    Get project root directory.
    
    Searches upward for markers like 'src' folder or 'setup.py'.
    """
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / 'src').exists() or (parent / 'setup.py').exists():
            return parent
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
