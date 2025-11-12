"""
Pytest configuration for test suite
"""

import sys
from pathlib import Path

# Add parent directory to path so tests can import project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
