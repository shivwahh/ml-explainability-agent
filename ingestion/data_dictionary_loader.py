"""
data_dictionary_loader.py

Ingestion loader for the data dictionary artifact.

Parses a data dictionary from ``.xlsx`` (with ``.csv`` fallback) into
normalized feature records that form the semantic layer used when
explaining individual features.
"""

from pathlib import Path

import pandas as pd


class DataDictionaryLoader:
    """
    Load and normalize a data dictionary into feature records.

    Column headers are matched against a set of known aliases and mapped
    to canonical keys. Unrecognized columns are preserved under their
    normalized header so no information is lost.
    """

    SUPPORTED_EXTENSIONS = {".xlsx", ".csv"}

    COLUMN_ALIASES = {
        "name": {
            "column",
            "column name",
            "feature",
            "feature name",
            "field",
            "variable",
            "name",
        },
        "meaning": {
            "meaning",
            "description",
            "definition",
        },
        "unit": {
            "unit",
            "units",
        },
        "business_description": {
            "business description",
            "business meaning",
            "business",
        },
        "allowed_range": {
            "allowed range",
            "range",
            "valid range",
            "value range",
        },
        "data_type": {
            "data type",
            "dtype",
            "type",
        },
    }

    def __init__(self, dictionary_path: str):
        """
        Args:
            dictionary_path: Path to the data dictionary file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        self.dictionary_path = Path(dictionary_path)

        if not self.dictionary_path.exists():
            raise FileNotFoundError(
                f"Data dictionary not found: {self.dictionary_path}"
            )

        suffix = self.dictionary_path.suffix.lower()

        if suffix not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported data dictionary extension: '{suffix}'. "
                f"Supported extensions: "
                f"{sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        self.dataframe = None

    def load(self) -> pd.DataFrame:
        """
        Load the data dictionary into a pandas DataFrame.

        Returns:
            The raw data dictionary as a DataFrame.

        Raises:
            ImportError: If the ``.xlsx`` engine (openpyxl) is missing.
        """
        suffix = self.dictionary_path.suffix.lower()

        if suffix == ".xlsx":
            try:
                self.dataframe = pd.read_excel(
                    self.dictionary_path,
                    engine="openpyxl",
                )
            except ImportError as error:
                raise ImportError(
                    "Reading .xlsx data dictionaries requires the "
                    "'openpyxl' package. Install it with "
                    "'pip install openpyxl' or provide a .csv file."
                ) from error
        else:
            self.dataframe = pd.read_csv(self.dictionary_path)

        return self.dataframe

    def _canonical_key(self, header: str) -> str:
        """
        Map a raw column header to its canonical key.

        Args:
            header: The original column header.

        Returns:
            The canonical key if a known alias matches, otherwise the
            normalized (lowercased, underscored) header.
        """
        normalized = str(header).strip().lower()

        for canonical, aliases in self.COLUMN_ALIASES.items():
            if normalized in aliases:
                return canonical

        return normalized.replace(" ", "_")

    def to_records(self) -> list:
        """
        Return normalized feature records.

        The dictionary is loaded automatically if it has not been loaded
        yet. Empty / missing cell values are represented as ``None``.

        Returns:
            A list of dictionaries, one per feature row.
        """
        if self.dataframe is None:
            self.load()

        column_mapping = {
            column: self._canonical_key(column)
            for column in self.dataframe.columns
        }

        records = []

        for _, row in self.dataframe.iterrows():
            record = {}

            for original, canonical in column_mapping.items():
                value = row[original]

                if pd.isna(value):
                    record[canonical] = None
                elif isinstance(value, str):
                    record[canonical] = value.strip()
                else:
                    record[canonical] = value

            records.append(record)

        return records
