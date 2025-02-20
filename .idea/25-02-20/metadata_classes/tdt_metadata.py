import json

from pydantic import BaseModel, Field


class MetadataStream(BaseModel):
    """Defines and validates the structure of metadata."""

    name: str
    code: str
    size: int
    type_: str
    type_str: str
    ucf: str
    fs: float
    num_channels: int
    dform: str
    start_time: float
    channel: list
    data_shape: tuple  # (num_channels, samples)

    @classmethod
    def from_json(cls, json_path: str):
        """Load metadata from a JSON file."""
        with open(json_path, "r") as file:
            metadata = json.load(file)
        return cls(**metadata)

    @classmethod
    def from_dict(cls, metadata_dict: dict):
        """Load metadata from a dictionary."""
        return cls(**metadata_dict)
