import pytest

from ingestion.data_dictionary_loader import DataDictionaryLoader


def _write_csv(tmp_path):
    path = tmp_path / "data_dictionary.csv"
    path.write_text(
        "Column,Meaning,Unit,Allowed Range\n"
        "Age,Customer Age,Years,18-100\n"
        "Income,Monthly Income,INR,>0\n",
        encoding="utf-8",
    )
    return path


def test_load_csv(tmp_path):
    path = _write_csv(tmp_path)

    loader = DataDictionaryLoader(str(path))
    frame = loader.load()

    assert len(frame) == 2


def test_to_records_normalizes_columns(tmp_path):
    path = _write_csv(tmp_path)

    records = DataDictionaryLoader(str(path)).to_records()

    assert records[0]["name"] == "Age"
    assert records[0]["meaning"] == "Customer Age"
    assert records[0]["unit"] == "Years"
    assert records[0]["allowed_range"] == "18-100"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DataDictionaryLoader("missing.csv")


def test_unsupported_extension_raises(tmp_path):
    bad_path = tmp_path / "dict.json"
    bad_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError):
        DataDictionaryLoader(str(bad_path))
