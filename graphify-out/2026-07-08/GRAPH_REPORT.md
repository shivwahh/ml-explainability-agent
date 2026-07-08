# Graph Report - .  (2026-07-07)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 508 nodes · 766 edges · 58 communities (40 shown, 18 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 32 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `a251a835`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_test_model_adapters.py|test_model_adapters.py]]
- [[_COMMUNITY_ModelArtifactLoader|ModelArtifactLoader]]
- [[_COMMUNITY_test_explainer_selector.py|test_explainer_selector.py]]
- [[_COMMUNITY_ModelLoader|ModelLoader]]
- [[_COMMUNITY_explainability_graph.py|explainability_graph.py]]
- [[_COMMUNITY_Artifact Consistency Validation|Artifact Consistency Validation]]
- [[_COMMUNITY_test_explanation_provider.py|test_explanation_provider.py]]
- [[_COMMUNITY_Intent Classification for Routing|Intent Classification for Routing]]
- [[_COMMUNITY_ProjectContextLoader|ProjectContextLoader]]
- [[_COMMUNITY_test_agent_runner.py|test_agent_runner.py]]
- [[_COMMUNITY_build_explanation_prompt|build_explanation_prompt]]
- [[_COMMUNITY_DataDictionaryLoader|DataDictionaryLoader]]
- [[_COMMUNITY_Project Configuration|Project Configuration]]
- [[_COMMUNITY_test_ingestion_node.py|test_ingestion_node.py]]
- [[_COMMUNITY_config_loader.py|config_loader.py]]
- [[_COMMUNITY_DecisionTreeReader|DecisionTreeReader]]
- [[_COMMUNITY_Python Project Requirements|Python Project Requirements]]
- [[_COMMUNITY_generate_demo_artifacts.py|generate_demo_artifacts.py]]
- [[_COMMUNITY_Enterprise ML Explainability Agent Vision|Enterprise ML Explainability Agent Vision]]
- [[_COMMUNITY_feature_importance_extractor.py|feature_importance_extractor.py]]
- [[_COMMUNITY_Explainability Engine Components|Explainability Engine Components]]
- [[_COMMUNITY_Model and Planner Tools|Model and Planner Tools]]
- [[_COMMUNITY_Feature Importance Visualization|Feature Importance Visualization]]
- [[_COMMUNITY_Decision Tree Reader Tool|Decision Tree Reader Tool]]
- [[_COMMUNITY_Explainer Agent Package|Explainer Agent Package]]
- [[_COMMUNITY_Planner Agent Package|Planner Agent Package]]
- [[_COMMUNITY_Report Agent Package|Report Agent Package]]
- [[_COMMUNITY_Visualization Agent Package|Visualization Agent Package]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY_ConfigLoader|ConfigLoader]]
- [[_COMMUNITY_DecisionPathExporter|DecisionPathExporter]]
- [[_COMMUNITY_ExplainabilityReportGenerator|ExplainabilityReportGenerator]]
- [[_COMMUNITY_FeatureImportanceExporter|FeatureImportanceExporter]]
- [[_COMMUNITY_LangGraph Explainability Agent|LangGraph Explainability Agent]]
- [[_COMMUNITY_ModelMetadataExtractor|ModelMetadataExtractor]]
- [[_COMMUNITY_TreeStructureExporter|TreeStructureExporter]]
- [[_COMMUNITY_Iris Species Classifier Project Context|Iris Species Classifier Project Context]]
- [[_COMMUNITY_ml-explainability-agent|ml-explainability-agent]]
- [[_COMMUNITY_ML Explainability Agent README|ML Explainability Agent README]]

## God Nodes (most connected - your core abstractions)
1. `ModelArtifactLoader` - 18 edges
2. `ProjectKnowledge` - 18 edges
3. `ArtifactValidator` - 14 edges
4. `ModelAdapter` - 14 edges
5. `SklearnTreeAdapter` - 14 edges
6. `ProjectContextLoader` - 13 edges
7. `IntentClassifier` - 12 edges
8. `build_explanation_prompt()` - 12 edges
9. `DataDictionaryLoader` - 11 edges
10. `GenericModelAdapter` - 11 edges

## Surprising Connections (you probably didn't know these)
- `test_load_config()` --calls--> `ConfigLoader`  [EXTRACTED]
  tests/test_config_loader.py → utils/config_loader.py
- `test_extract_path()` --calls--> `DecisionPathExtractor`  [EXTRACTED]
  tests/test_decision_path_extractor.py → tools/tree_reader/decision_path_extractor.py
- `test_tree_summary()` --calls--> `DecisionTreeReader`  [EXTRACTED]
  tests/test_decision_tree_reader.py → tools/tree_reader/decision_tree_reader.py
- `test_save_and_load_model()` --calls--> `ModelLoader`  [EXTRACTED]
  tests/test_model_loader.py → tools/model_loader/model_loader.py
- `Model Loader Tool` --implements--> `ModelLoader`  [INFERRED]
  docs/architecture.md → AGENTS.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Demo Configuration Sections** — configs_demo_config_ingestion, configs_demo_config_routing, configs_demo_config_explanation [INFERRED 0.75]
- **Project Configuration Sections** — configs_project_config_project, configs_project_config_ingestion, configs_project_config_routing, configs_project_config_explanation, configs_project_config_data, configs_project_config_model, configs_project_config_explainability, configs_project_config_visualization, configs_project_config_logging, configs_project_config_agent, configs_project_config_output [INFERRED 0.75]
- **Core Libraries for ML Explainability Agent** — requirements_txt_numpy, requirements_txt_pandas, requirements_txt_scikit_learn [INFERRED 0.80]
- **Core Project Inputs** — context_model_artifact, context_data_dictionary, context_project_context [EXTRACTED 0.95]
- **Internal Knowledge Layers** — context_project_knowledge, context_internal_agent_knowledge_graph, context_enterprise_ml_explainability_agent [INFERRED 0.80]
- **Explainability Tools Group** — agents_decisiontreereader, agents_decisionpathextractor, agents_featureimportanceextractor, tools_treereadertool, tools_shaptool, agents_explainabilityengine [INFERRED 0.85]

## Communities (58 total, 18 thin omitted)

### Community 0 - "test_model_adapters.py"
Cohesion: 0.06
Nodes (38): ModelAdapter, base.py  Base model adapter interface.  Defines the :class:`ModelAdapter` co, Extract class labels for classifiers when available., Extract the number of input features when available., Base class for framework-specific model adapters.      Subclasses declare the, Return whether this adapter can handle the given estimator.          Args:, Extract normalized metadata for the given model.          Args:             m, Infer whether the estimator is a classifier or regressor.          Returns: (+30 more)

### Community 1 - "ModelArtifactLoader"
Cohesion: 0.08
Nodes (30): ingestion  Ingestion layer for the ML Explainability Agent.  Loads, normaliz, ModelArtifactLoader, model_loader.py  Ingestion loader for trained model artifacts.  Loads ``.pkl, Load a model artifact and extract normalized metadata.      The loader support, Args:             model_path: Path to the serialized model artifact., Load and return the deserialized model object.          Returns:, Return the underlying estimator, unwrapping a Pipeline.          Returns:, Extract normalized metadata from the loaded model.          The model is loade (+22 more)

### Community 2 - "test_explainer_selector.py"
Cohesion: 0.09
Nodes (28): test_extract_path(), _forest(), _iris(), test_feature_importance_selector_works_for_both(), test_forest_path_aggregates_across_estimators(), test_is_ensemble_falls_back_to_attribute(), test_is_ensemble_from_metadata(), test_selects_decision_path_for_single_tree() (+20 more)

### Community 3 - "ModelLoader"
Cohesion: 0.07
Nodes (15): Args:             dictionary_path: Path to the data dictionary file., Path, test_save_and_load_model(), ModelLoader, model_loader.py  Utility for saving, loading and inspecting machine learning, Handles model persistence operations., Load model from disk., Return all saved models. (+7 more)

### Community 4 - "explainability_graph.py"
Cohesion: 0.13
Nodes (24): decision_path_node(), feature_importance_node(), _feature_names(), _model_metadata(), planner_node(), prediction_node(), Return feature names from the knowledge object.      Prefers the model's own f, Return the normalized model metadata from the knowledge object. (+16 more)

### Community 5 - "Artifact Consistency Validation"
Cohesion: 0.14
Nodes (17): ArtifactValidator, artifact_validator.py  Validation for ingested artifacts.  Combines the norm, Cross-check model features against the data dictionary., Validate ingested artifacts and produce a structured report.      The validato, Run all validations and return a structured report.          Returns:, Args:             model_metadata: Output of                 :meth:`ModelArtifa, Return the set of feature names declared in the data dictionary., Validate the model metadata section. (+9 more)

### Community 6 - "test_explanation_provider.py"
Cohesion: 0.15
Nodes (18): build_provider(), EchoExplanationProvider, ExplanationProvider, OpenAIExplanationProvider, explanation_provider.py  Config-driven explanation providers.  Decouples the, Base interface for explanation text generation., Return explanation text for the given prompt.          Args:             prom, Deterministic provider that echoes the prompt.      Used as the default and fo (+10 more)

### Community 7 - "Intent Classification for Routing"
Cohesion: 0.13
Nodes (18): _classifier(), _intents(), test_classifies_decision_path(), test_classifies_feature_importance(), test_classifies_prediction(), test_empty_question_falls_back_to_default(), test_first_matching_intent_wins(), test_unknown_question_falls_back_to_default() (+10 more)

### Community 8 - "ProjectContextLoader"
Cohesion: 0.13
Nodes (16): ProjectContextLoader, project_context_loader.py  Ingestion loader for the project context artifact., Read a plain text or markdown file as UTF-8 text., Load unstructured project context and split it into sections.      Readers are, Args:             context_path: Path to the project context file.          Ra, Register a reader for a file extension.          Args:             extension:, Return the currently supported extensions., Load and return the raw project context text.          Returns:             T (+8 more)

### Community 9 - "test_agent_runner.py"
Cohesion: 0.20
Nodes (15): build_initial_state(), agent_runner.py  End-to-end runner for the explainability agent graph.  Asse, Assemble the initial agent state from config and inputs.      Args:         c, Run the compiled explainability graph end to end.      Args:         config_p, run_agent(), main(), _parse_sample(), main.py  Command-line entry point for the ML Explainability Agent.  Runs the ful (+7 more)

### Community 10 - "build_explanation_prompt"
Cohesion: 0.19
Nodes (16): build_explanation_prompt(), _feature_meanings(), explanation_prompt.py  Business-context-aware explanation prompt builder.  C, Render project-context sections when available., Build the business-context-aware explanation prompt.      Args:         state, Build a map of feature name to a human-readable meaning string.      Combines, Return the feature names referenced by the explanation signals.      Handles b, Render meanings for the features referenced in the explanation. (+8 more)

### Community 11 - "DataDictionaryLoader"
Cohesion: 0.18
Nodes (12): DataFrame, DataDictionaryLoader, data_dictionary_loader.py  Ingestion loader for the data dictionary artifact., Map a raw column header to its canonical key.          Args:             head, Return normalized feature records.          The dictionary is loaded automatic, Load and normalize a data dictionary into feature records.      Column headers, Load the data dictionary into a pandas DataFrame.          Returns:, test_load_csv() (+4 more)

### Community 12 - "Project Configuration"
Cohesion: 0.14
Nodes (16): Demo Configuration, Demo Explanation Settings, Demo Ingestion Settings, Demo Routing Settings, Project Configuration, Agent Framework Configuration, Data Paths Configuration, Explainability Methods Configuration (+8 more)

### Community 13 - "test_ingestion_node.py"
Cohesion: 0.25
Nodes (11): ingestion_node(), ingestion_node.py  LangGraph entry node that builds the unified ProjectKnowled, Build the project knowledge object and populate agent state.      Reads the mo, ExplainabilityState, Shared state for the explainability agent graph.      ``config_path`` drives i, _make_model(), test_ingestion_node_handles_missing_artifacts(), test_ingestion_node_populates_knowledge_and_model() (+3 more)

### Community 14 - "config_loader.py"
Cohesion: 0.19
Nodes (7): test_load_config(), ConfigLoader, config_loader.py  Utility functions for loading project configuration files., Loads configuration from YAML files., Read YAML configuration file., Retrieve nested configuration values.          Example:         config.get("m, Return complete configuration.

### Community 15 - "DecisionTreeReader"
Cohesion: 0.20
Nodes (5): test_tree_summary(), DecisionTreeReader, decision_tree_reader.py  Utilities for understanding and explaining Decision, Return basic tree information., Extract all split rules.

### Community 16 - "Python Project Requirements"
Cohesion: 0.18
Nodes (11): graphviz, jupyter, matplotlib, numpy, openpyxl, pandas, pytest, python-dotenv (+3 more)

### Community 17 - "generate_demo_artifacts.py"
Cohesion: 0.25
Nodes (10): main(), generate_demo_artifacts.py  Create a self-contained set of demo artifacts for, Train and persist a fitted decision tree on the Iris dataset., Write a data dictionary describing the Iris features., Write a short project context markdown file., Write a demo configuration pointing at the generated artifacts., _train_model(), _write_config() (+2 more)

### Community 18 - "Enterprise ML Explainability Agent Vision"
Cohesion: 0.28
Nodes (9): Data Dictionary, Step-by-Step Development Roadmap, Enterprise ML Explainability Agent Vision, Internal Agent Knowledge Graph, Model Artifact, Project Context, Project Knowledge Object, Modular Repository Structure (+1 more)

### Community 19 - "feature_importance_extractor.py"
Cohesion: 0.28
Nodes (4): FeatureImportanceExtractor, feature_importance_extractor.py  Extract feature importance from Decision Tre, Return feature importance dataframe., Return top N important features.

### Community 20 - "Explainability Engine Components"
Cohesion: 0.33
Nodes (6): DecisionPathExtractor, Explainability Engine, FeatureImportanceExtractor, Report Generator, SHAP Tool, Visualization Tool

### Community 21 - "Model and Planner Tools"
Cohesion: 0.40
Nodes (5): ModelLoader, Planner Agent, Metrics Tool, Model Loader Tool, Prediction Tool

## Knowledge Gaps
- **38 isolated node(s):** `ml-explainability-agent`, `ConfigLoader`, `ModelLoader`, `DecisionTreeReader`, `DecisionPathExtractor` (+33 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **18 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModelArtifactLoader` connect `ModelArtifactLoader` to `test_model_adapters.py`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Why does `ProjectKnowledge` connect `ModelArtifactLoader` to `test_ingestion_node.py`, `Intent Classification for Routing`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `ModelAdapter` (e.g. with `GenericModelAdapter` and `SklearnTreeAdapter`) actually correct?**
  _`ModelAdapter` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `SklearnTreeAdapter` (e.g. with `AdapterRegistry` and `ModelAdapter`) actually correct?**
  _`SklearnTreeAdapter` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Explainer agent package.`, `Planner agent package.`, `Report agent package.` to the rest of the system?**
  _177 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_model_adapters.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06291591046581972 - nodes in this community are weakly interconnected._
- **Should `ModelArtifactLoader` be split into smaller, more focused modules?**
  _Cohesion score 0.07928118393234672 - nodes in this community are weakly interconnected._