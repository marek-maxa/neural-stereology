import sys
from pathlib import Path

LAMELLA_ROOT = Path(__file__).resolve().parents[1] / "src" / "lamella"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(LAMELLA_ROOT))
