import sys
from pathlib import Path

# Make evaluation/ (common.py, eval_*.py) and tests/ importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
