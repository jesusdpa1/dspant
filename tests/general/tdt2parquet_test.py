# %%
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import dotenv
import tdt

from dspant.io import convert_tdt_to_ant

dotenv.load_dotenv()
# %%
home = Path(os.getenv("DATA_DIR"))
# %% Set up paths


def ls(directory: Path):
    entries = [entry for entry in directory.iterdir()]
    for i, entry in enumerate(entries, start=1):
        entry_type = "File" if entry.is_file() else "Directory"
        print(f"{i}. {entry.name} ({entry_type})")


tank_path = home.joinpath(r"E:\jpenalozaa\test_")

ls(tank_path)
# %%

block_path = tank_path.joinpath("00_baseline")
convert_tdt_to_ant(block_path, start=0, end=-1)

# %%

# tdt_block = tdt.read_block(str(block_path), t1=0, t2=1)

# %%
