"""Start the local demo using the current Python, or an optional local UI environment."""
import json
import os
from pathlib import Path
import sys


root = Path(__file__).resolve().parent
config_file = root / "local_config.json"
config = json.loads(config_file.read_text()) if config_file.is_file() else {}
python = os.environ.get("SAX_NERF_DEMO_PYTHON", config.get("ui_python", sys.executable))
os.execv(python, [python, str(root / "app.py"), *sys.argv[1:]])
