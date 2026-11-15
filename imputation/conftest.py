import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for sub in (ROOT, ROOT / "scripts"):
    if str(sub) not in sys.path:
        sys.path.insert(0, str(sub))
