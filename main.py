"""
main.py

Command-line entry point for the ML Explainability Agent.

Runs the full LangGraph pipeline (ingestion -> routing -> explanation)
on a question and optional sample, driven entirely by configuration.

Example:
    python main.py --config configs/demo_config.yaml \
        --question "Why this species?" --sample "5.1,3.5,1.4,0.2"
"""

import argparse

from agents.agent_runner import run_agent


def _parse_sample(raw):
    """Parse a comma-separated sample string into a list of floats."""
    if not raw:
        return None

    return [float(value) for value in raw.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Run the ML explainability agent."
    )
    parser.add_argument(
        "--config",
        default="configs/demo_config.yaml",
        help="Path to the project configuration YAML.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The question to ask about the model or a prediction.",
    )
    parser.add_argument(
        "--sample",
        default=None,
        help="Comma-separated feature values for the record to explain.",
    )

    args = parser.parse_args()

    result = run_agent(
        args.config,
        args.question,
        _parse_sample(args.sample),
    )

    if result.get("explanation"):
        print(result["explanation"])
        return

    for key in ("prediction", "decision_path", "feature_importance", "local_explanation"):
        if key in result:
            print(f"{key}: {result[key]}")


if __name__ == "__main__":
    main()

