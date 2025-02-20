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

    @field_validator("data_path")
    def validate_data_path(cls, value):
        path_ = Path(value)
        if not path_.suffix == ".ant":
            raise ValueError("Data path must have a .ant extension")
        return str(path_.resolve())

    def validate_files(self):
        """Ensure required data and metadata files exist"""
        path_ = Path(self.data_path)
        self.parquet_path = path_ / f"data_{path_.stem}.parquet"
        self.metadata_path = path_ / f"metadata_{path_.stem}.json"

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.parquet_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

    def load_metadata(self):
        """Load metadata from file"""
        self.validate_files()
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        base_metadata = metadata.get("base", {})
        for key, value in base_metadata.items():
            setattr(self, key, value)

        self.metadata = metadata
