import sys
from pathlib import Path


LAMELLA_ROOT = Path(__file__).resolve().parents[1] / "src" / "lamella"
sys.path.insert(0, str(LAMELLA_ROOT))
