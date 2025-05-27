import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
FIGURE_TITLE = "ecg_template subtraction"
FIGURE_DIR = Path(os.getenv("FIGURE_DIR"))
FIGURE_PATH = FIGURE_DIR.joinpath(f"{FIGURE_TITLE}.png")
