import json
from pathlib import Path
from typing import Any, Dict, Optional, Self

from pydantic import BaseModel, ConfigDict, Field  # , field_validator

from dspant.core.internals import public_api


@public_api
class BaseNode(BaseModel):
    """
    Base class for handling data paths and metadata in dspant.

    This class provides the foundation for data node types like StreamNode and EpocNode.
    It handles path validation and metadata loading.
    """

    data_path: str = Field(..., description="Parent folder for data storage")
    parquet_path: Optional[str] = None
    metadata_path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")

    def validate_files(self) -> None:
        """
        Ensure required data and metadata files exist.
        Sets parquet_path and metadata_path if files are found.

        Raises:
            FileNotFoundError: If required data or metadata files are not found
        """
        import glob

        path_ = Path(self.data_path)

        # Use glob to find matching files
        data_pattern = str(path_ / f"data_*.parquet")
        metadata_pattern = str(path_ / f"metadata_*.json")

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

    def load_metadata(self) -> Self:
        """
        Load metadata from file and apply base attributes to this instance.

        Returns:
            Self: Returns self for method chaining

        Raises:
            FileNotFoundError: If metadata file doesn't exist
            json.JSONDecodeError: If metadata file contains invalid JSON
        """
        self.validate_files()
        try:
            with open(self.metadata_path, "r") as f:
                metadata = json.load(f)

            base_metadata = metadata.get("base", {})
            for key, value in base_metadata.items():
                setattr(self, key, value)

            self.metadata = metadata
            return self
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Invalid JSON in metadata file {self.metadata_path}: {str(e)}",
                e.doc,
                e.pos,
            )
