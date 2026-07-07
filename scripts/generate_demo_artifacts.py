"""
generate_demo_artifacts.py

Create a self-contained set of demo artifacts for the explainability
playground (UI and notebook):

* a fitted scikit-learn decision tree saved as a ``.joblib`` model,
* a data dictionary describing its features,
* a project context markdown file, and
* a demo configuration wiring the three artifacts together.

Run:
    python scripts/generate_demo_artifacts.py
"""

from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "demo_iris_tree.joblib"
DEMO_DIR = ROOT / "data" / "demo"
DICTIONARY_PATH = DEMO_DIR / "data_dictionary.csv"
CONTEXT_PATH = DEMO_DIR / "project_context.md"
CONFIG_PATH = ROOT / "configs" / "demo_config.yaml"


def _train_model():
    """Train and persist a fitted decision tree on the Iris dataset."""
    data = load_iris()

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(data.data, data.target)
    model.feature_names_in_ = data.feature_names

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return data


def _write_data_dictionary(data):
    """Write a data dictionary describing the Iris features."""
    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        ("sepal length (cm)", "Length of the sepal", "cm", "0-10"),
        ("sepal width (cm)", "Width of the sepal", "cm", "0-10"),
        ("petal length (cm)", "Length of the petal", "cm", "0-10"),
        ("petal width (cm)", "Width of the petal", "cm", "0-10"),
    ]

    lines = ["Column,Meaning,Unit,Allowed Range"]
    lines += [",".join(row) for row in rows]

    DICTIONARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_context():
    """Write a short project context markdown file."""
    CONTEXT_PATH.write_text(
        "# Project Name\n\n"
        "Iris Species Classifier\n\n"
        "# Business Objective\n\n"
        "Classify an iris flower into one of three species from its "
        "sepal and petal measurements.\n",
        encoding="utf-8",
    )


def _write_config():
    """Write a demo configuration pointing at the generated artifacts."""
    CONFIG_PATH.write_text(
        "# Demo configuration for the explainability playground.\n\n"
        "ingestion:\n"
        f'  model_path: "{MODEL_PATH.as_posix()}"\n'
        f'  data_dictionary_path: "{DICTIONARY_PATH.as_posix()}"\n'
        f'  project_context_path: "{CONTEXT_PATH.as_posix()}"\n\n'
        "routing:\n"
        '  default_intent: "full_explanation"\n'
        "  intents:\n"
        "    decision_path:\n"
        '      - "decision path"\n'
        '      - "why did"\n'
        '      - "how did"\n'
        '      - "which rules"\n'
        '      - "path"\n'
        "    feature_importance:\n"
        '      - "feature importance"\n'
        '      - "most important"\n'
        '      - "which features"\n'
        '      - "influential"\n'
        '      - "drivers"\n'
        "    prediction:\n"
        '      - "what will"\n'
        '      - "predict"\n'
        '      - "prediction"\n'
        '      - "outcome"\n\n'
        "explanation:\n"
        '  provider: "echo"\n'
        '  model: "gpt-4.1-mini"\n',
        encoding="utf-8",
    )


def main():
    data = _train_model()
    _write_data_dictionary(data)
    _write_context()
    _write_config()

    print(f"Model:          {MODEL_PATH}")
    print(f"Data dictionary:{DICTIONARY_PATH}")
    print(f"Project context:{CONTEXT_PATH}")
    print(f"Demo config:    {CONFIG_PATH}")


if __name__ == "__main__":
    main()
