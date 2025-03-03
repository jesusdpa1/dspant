import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseNode(BaseModel):
    """Base class for handling data paths and metadata"""

    data_path: str = Field(..., description="Parent folder for data storage")
    parquet_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")

    def validate_files(self):
        """Ensure required data and metadata files exist"""
        import glob

        path_ = Path(self.data_path)

        # Use glob to find matching files
        data_pattern = str(path_ / f"data_{path_.stem}*.parquet")
        metadata_pattern = str(path_ / f"metadata_{path_.stem}*.json")

        data_files = glob.glob(data_pattern)
        metadata_files = glob.glob(metadata_pattern)

        if not data_files:
            raise FileNotFoundError(f"No data file found matching: {data_pattern}")
        if not metadata_files:
            raise FileNotFoundError(
                f"No metadata file found matching: {metadata_pattern}"
            )

        # Use the first matching file (or you could implement logic to choose a specific one)
        self.parquet_path = data_files[0]
        self.metadata_path = metadata_files[0]

    def load_metadata(self):
        """Load metadata from file"""
        self.validate_files()
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        base_metadata = metadata.get("base", {})
        for key, value in base_metadata.items():
            setattr(self, key, value)

        self.metadata = metadata
