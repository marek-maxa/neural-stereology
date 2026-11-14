import sys
from pathlib import Path

# The inherited lamella generator is not an installable package; its modules
# (``main``, ``core.*``) import each other relative to ``src/lamella``.
LAMELLA_ROOT = Path(__file__).resolve().parents[1] / "src" / "lamella"
sys.path.insert(0, str(LAMELLA_ROOT))
