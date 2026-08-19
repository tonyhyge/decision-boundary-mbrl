import sys
from pathlib import Path

# Ensure root directory is always on Python path for pytest
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
