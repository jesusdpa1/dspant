from typing import Optional

import polars as pl
from rich.console import Console
from rich.table import Table

from dspant.core.internals import public_api
from dspant.nodes.base import BaseNode


class BaseEpocNode(BaseNode):
    """Base class for handling event-based data"""

    name: Optional[str] = None


@public_api
class EpocNode(BaseEpocNode):
    """Class for loading and accessing epoch data"""

    def __init__(self, data_path: str, **kwargs):
        super().__init__(data_path=data_path, **kwargs)
        self.data = None

    def load_data(self, force_reload: bool = False) -> pl.DataFrame:
        """Load data into a Polars DataFrame"""
        if self.data is not None and not force_reload:
            return self.data

        try:
            # Ensure files are validated before loading
            if self.parquet_path is None:
                self.validate_files()

            self.data = pl.read_parquet(str(self.parquet_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load epoch data: {e}") from e

        return self.data

    def summarize(self):
        """Print a summary of the epoch node configuration and metadata"""
        console = Console()

        # Create main table
        table = Table(title="Epoch Node Summary")
        table.add_column("Attribute", justify="right", style="cyan")
        table.add_column("Value", justify="left")

        # Add file information
        table.add_section()
        table.add_row("Data Path", str(self.data_path))
        table.add_row(
            "Parquet Path",
            str(self.parquet_path) if self.parquet_path else "Not validated",
        )
        table.add_row(
            "Metadata Path",
            str(self.metadata_path) if self.metadata_path else "Not validated",
        )

        # Add metadata information
        if self.metadata:
            table.add_section()
            table.add_row("Name", str(self.name))

            # Add any other metadata fields from base
            for key, value in self.metadata.get("base", {}).items():
                if key not in ["data_path", "metadata", "name"]:
                    table.add_row(key, str(value))

        # Add data information if loaded
        if self.data is not None:
            table.add_section()
            table.add_row("Number of Events", str(len(self.data)))
            table.add_row("Columns", ", ".join(self.data.columns))
            table.add_row("Data Schema", str(self.data.schema))

        console.print(table)
