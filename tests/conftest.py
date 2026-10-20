import sys
from pathlib import Path

# Let the tests import pilot.py, analyse.py and prompts.py from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
