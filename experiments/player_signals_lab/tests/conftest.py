import sys
from pathlib import Path

sys.dont_write_bytecode = True
LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB.parents[1] / "src"))
sys.path.insert(0, str(LAB))
