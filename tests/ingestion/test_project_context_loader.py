import pytest

from ingestion.project_context_loader import ProjectContextLoader


def _write_markdown(tmp_path):
    path = tmp_path / "project_context.md"
    path.write_text(
        "# Project Name\n\n"
        "Customer Churn Prediction\n\n"
        "# Business Objective\n\n"
        "Reduce churn.\n",
        encoding="utf-8",
    )
    return path


def test_load_markdown(tmp_path):
    path = _write_markdown(tmp_path)

    text = ProjectContextLoader(str(path)).load()

    assert "Customer Churn Prediction" in text


def test_to_sections(tmp_path):
    path = _write_markdown(tmp_path)

    sections = ProjectContextLoader(str(path)).to_sections()

    assert sections["Project Name"] == "Customer Churn Prediction"
    assert sections["Business Objective"] == "Reduce churn."


def test_register_reader_extends_support(tmp_path):
    path = tmp_path / "context.log"
    path.write_text("plain content", encoding="utf-8")

    loader = ProjectContextLoader(str(path))
    loader.register_reader(
        ".log",
        lambda p: p.read_text(encoding="utf-8"),
    )

    assert loader.load() == "plain content"


def test_unregistered_extension_raises(tmp_path):
    path = tmp_path / "context.pdf"
    path.write_text("binary-ish", encoding="utf-8")

    with pytest.raises(ValueError):
        ProjectContextLoader(str(path)).load()


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ProjectContextLoader("missing.md")
