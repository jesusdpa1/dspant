# %%
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import tdt

from dspant.io import convert_tdt_to_ant

# %% Set up paths


def ls(directory: Path):
    entries = [entry for entry in directory.iterdir()]
    for i, entry in enumerate(entries, start=1):
        entry_type = "File" if entry.is_file() else "Directory"
        print(f"{i}. {entry.name} ({entry_type})")


home = Path(r"E:\jpenalozaa")  # Path().home()
tank_path = home.joinpath(r"topoMapping\25-03-22_4896-2_testSubject_topoMapping")
# E:\jpenalozaa\\
ls(tank_path)
# %%

block_path = tank_path.joinpath("18-21-42_meps")
convert_tdt_to_ant(block_path, start=0, end=-1)

# %%
