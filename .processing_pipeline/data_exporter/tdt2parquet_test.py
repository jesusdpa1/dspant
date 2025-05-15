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
    path_list = []
    entries = [entry for entry in directory.iterdir()]
    for i, entry in enumerate(entries, start=1):
        entry_type = "File" if entry.is_file() else "Directory"
        print(f"{i}. {entry.name} ({entry_type})")
        if entry_type == "Directory":
            path_list.append(entry)

    return path_list


tank_path = home.joinpath(r"topoMapping\25-03-16_4896-1_testSubject_topoMapping")
# output_path = tank_path.joinpath("drv")
a = ls(tank_path)
filtered_paths = [entry for entry in a if entry.name != "drv"]

# ls(output_path)

# %%
for block_path in filtered_paths:
    convert_tdt_to_ant(block_path, output_path=tank_path, start=0, end=-1)
print("done")

# %%

# tdt_block = tdt.read_block(str(block_path), t1=0, t2=1)

# %%
