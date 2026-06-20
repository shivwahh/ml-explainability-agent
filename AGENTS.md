# ML Explainability Agent

## Project Goal

Build an enterprise-grade explainability platform capable of understanding and explaining:

* Decision Trees
* Random Forests
* XGBoost
* Forecasting Models
* Optimization Models

## Architecture Rules

* Configuration-driven architecture
* No hardcoded paths
* No hardcoded model assumptions
* All functionality implemented as tools
* Agents orchestrate tools
* Every module must have tests

## Folder Responsibilities

tools/

* Pure functionality
* No LLM calls

agents/

* LangGraph agents
* Tool orchestration

configs/

* YAML configuration

tests/

* Unit tests for every tool

## Coding Standards

* Type hints preferred
* Google-style docstrings
* Small reusable classes
* No notebook code copied into production code

## Current Phase

Phase 1: Decision Tree Explainability Agent

Current completed components:

* ConfigLoader
* ModelLoader
* DecisionTreeReader
* DecisionPathExtractor
* FeatureImportanceExtractor
* TreeStructureExporter

Next priorities:

1. ModelMetadataExtractor
2. DecisionPathExporter
3. FeatureImportanceExporter
4. ExplainabilityReportGenerator
5. LangGraph Explainability Agent
