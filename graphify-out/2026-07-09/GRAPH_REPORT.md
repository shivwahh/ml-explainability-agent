# Graph Report - .  (2026-07-09)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 579 nodes · 865 edges · 76 communities (46 shown, 30 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0d5d3328`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_test_explainer_selector.py|test_explainer_selector.py]]
- [[_COMMUNITY_ModelArtifactLoader|ModelArtifactLoader]]
- [[_COMMUNITY_ModelLoader|ModelLoader]]
- [[_COMMUNITY_explainability_graph.py|explainability_graph.py]]
- [[_COMMUNITY_test_explanation_provider.py|test_explanation_provider.py]]
- [[_COMMUNITY_Artifact Consistency Validation|Artifact Consistency Validation]]
- [[_COMMUNITY_ProjectContextLoader|ProjectContextLoader]]
- [[_COMMUNITY_test_intent_classifier.py|test_intent_classifier.py]]
- [[_COMMUNITY_app.py|app.py]]
- [[_COMMUNITY_test_agent_runner.py|test_agent_runner.py]]
- [[_COMMUNITY_test_model_adapters.py|test_model_adapters.py]]
- [[_COMMUNITY_build_explanation_prompt|build_explanation_prompt]]
- [[_COMMUNITY_DataDictionaryLoader|DataDictionaryLoader]]
- [[_COMMUNITY_Project Configuration|Project Configuration]]
- [[_COMMUNITY_ModelAdapter|ModelAdapter]]
- [[_COMMUNITY_decision_path_extractor.py|decision_path_extractor.py]]
- [[_COMMUNITY_XGBoostAdapter|XGBoostAdapter]]
- [[_COMMUNITY_ConfigLoader|ConfigLoader]]
- [[_COMMUNITY_test_ingestion_node.py|test_ingestion_node.py]]
- [[_COMMUNITY_sklearn_tree.py|sklearn_tree.py]]
- [[_COMMUNITY_DecisionTreeReader|DecisionTreeReader]]
- [[_COMMUNITY_generate_demo_artifacts.py|generate_demo_artifacts.py]]
- [[_COMMUNITY_Enterprise ML Explainability Agent Vision|Enterprise ML Explainability Agent Vision]]
- [[_COMMUNITY_Census Income (50K) Classifier|Census Income (>50K) Classifier]]
- [[_COMMUNITY_AdapterRegistry|AdapterRegistry]]
- [[_COMMUNITY_._booster_details|._booster_details]]
- [[_COMMUNITY_Explainability Engine|Explainability Engine]]
- [[_COMMUNITY_Planner Agent|Planner Agent]]
- [[_COMMUNITY_.extract|.extract]]
- [[_COMMUNITY_feature_importance_visualizer.py|feature_importance_visualizer.py]]
- [[_COMMUNITY_DecisionTreeReader|DecisionTreeReader]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
- [[_COMMUNITY___init__.py|__init__.py]]
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
- [[_COMMUNITY_graphviz|graphviz]]
- [[_COMMUNITY_jupyter|jupyter]]
- [[_COMMUNITY_matplotlib|matplotlib]]
- [[_COMMUNITY_numpy|numpy]]
- [[_COMMUNITY_openpyxl|openpyxl]]
- [[_COMMUNITY_pandas|pandas]]
- [[_COMMUNITY_pytest|pytest]]
- [[_COMMUNITY_python-dotenv|python-dotenv]]
- [[_COMMUNITY_scikit-learn|scikit-learn]]
- [[_COMMUNITY_streamlit|streamlit]]
- [[_COMMUNITY_xgboost|xgboost]]

## God Nodes (most connected - your core abstractions)
1. `ModelArtifactLoader` - 18 edges
2. `ProjectKnowledge` - 16 edges
3. `ModelAdapter` - 16 edges
4. `XGBoostAdapter` - 16 edges
5. `ArtifactValidator` - 14 edges
6. `default_registry()` - 14 edges
7. `ProjectContextLoader` - 13 edges
8. `XGBoostFeatureImportanceExtractor` - 13 edges
9. `build_explanation_prompt()` - 12 edges
10. `DataDictionaryLoader` - 11 edges

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
- **Machine Learning Project Context Documents** — data_adult_income_project_context_census_income_classifier, data_breast_cancer_project_context_breast_tumour_classifier, data_california_housing_project_context_california_house_value_regressor [INFERRED 0.90]
- **Core Data Science Stack** — requirements_numpy, requirements_pandas, requirements_openpyxl [INFERRED 0.80]
- **Visualization Stack** — requirements_matplotlib, requirements_graphviz [INFERRED 0.80]
- **Machine Learning Stack** — requirements_scikit_learn, requirements_xgboost [INFERRED 0.80]
- **Demo Configuration Sections** — configs_demo_config_ingestion, configs_demo_config_routing, configs_demo_config_explanation [INFERRED 0.75]
- **Project Configuration Sections** — configs_project_config_project, configs_project_config_ingestion, configs_project_config_routing, configs_project_config_explanation, configs_project_config_data, configs_project_config_model, configs_project_config_explainability, configs_project_config_visualization, configs_project_config_logging, configs_project_config_agent, configs_project_config_output [INFERRED 0.75]
- **Core Project Inputs** — context_model_artifact, context_data_dictionary, context_project_context [EXTRACTED 0.95]
- **Internal Knowledge Layers** — context_project_knowledge, context_internal_agent_knowledge_graph, context_enterprise_ml_explainability_agent [INFERRED 0.80]
- **Explainability Tools Group** — agents_decisiontreereader, agents_decisionpathextractor, agents_featureimportanceextractor, tools_treereadertool, tools_shaptool, agents_explainabilityengine [INFERRED 0.85]

## Communities (76 total, 30 thin omitted)

### Community 0 - "test_explainer_selector.py"
Cohesion: 0.06
Nodes (42): _forest(), _iris(), test_feature_importance_selector_works_for_both(), test_forest_path_aggregates_across_estimators(), test_is_ensemble_falls_back_to_attribute(), test_is_ensemble_from_metadata(), test_selects_decision_path_for_single_tree(), test_selects_forest_path_for_ensemble() (+34 more)

### Community 1 - "ModelArtifactLoader"
Cohesion: 0.08
Nodes (31): ingestion_node.py  LangGraph entry node that builds the unified ProjectKnowled, ingestion  Ingestion layer for the ML Explainability Agent.  Loads, normaliz, ModelArtifactLoader, model_loader.py  Ingestion loader for trained model artifacts.  Loads ``.pkl, Load a model artifact and extract normalized metadata.      The loader support, Args:             model_path: Path to the serialized model artifact., Load and return the deserialized model object.          Returns:, Return the underlying estimator, unwrapping a Pipeline.          Returns: (+23 more)

### Community 2 - "ModelLoader"
Cohesion: 0.07
Nodes (15): Args:             dictionary_path: Path to the data dictionary file., Path, test_save_and_load_model(), ModelLoader, model_loader.py  Utility for saving, loading and inspecting machine learning, Handles model persistence operations., Load model from disk., Return all saved models. (+7 more)

### Community 3 - "explainability_graph.py"
Cohesion: 0.13
Nodes (25): decision_path_node(), explanation_node(), feature_importance_node(), _feature_names(), _model_metadata(), planner_node(), prediction_node(), Return feature names from the knowledge object.      Prefers the model's own f (+17 more)

### Community 4 - "test_explanation_provider.py"
Cohesion: 0.14
Nodes (19): build_provider(), EchoExplanationProvider, ExplanationProvider, OpenAIExplanationProvider, explanation_provider.py  Config-driven explanation providers.  Decouples the, Build an explanation provider from configuration.      Reads ``explanation.pro, Point stale CA-bundle environment variables at a valid bundle.      Tools such, Base interface for explanation text generation. (+11 more)

### Community 5 - "Artifact Consistency Validation"
Cohesion: 0.14
Nodes (17): ArtifactValidator, artifact_validator.py  Validation for ingested artifacts.  Combines the norm, Cross-check model features against the data dictionary., Validate ingested artifacts and produce a structured report.      The validato, Run all validations and return a structured report.          Returns:, Args:             model_metadata: Output of                 :meth:`ModelArtifa, Return the set of feature names declared in the data dictionary., Validate the model metadata section. (+9 more)

### Community 6 - "ProjectContextLoader"
Cohesion: 0.13
Nodes (16): ProjectContextLoader, project_context_loader.py  Ingestion loader for the project context artifact., Read a plain text or markdown file as UTF-8 text., Load unstructured project context and split it into sections.      Readers are, Args:             context_path: Path to the project context file.          Ra, Register a reader for a file extension.          Args:             extension:, Return the currently supported extensions., Load and return the raw project context text.          Returns:             T (+8 more)

### Community 7 - "test_intent_classifier.py"
Cohesion: 0.16
Nodes (14): _classifier(), _intents(), test_classifies_decision_path(), test_classifies_feature_importance(), test_classifies_prediction(), test_empty_question_falls_back_to_default(), test_first_matching_intent_wins(), test_unknown_question_falls_back_to_default() (+6 more)

### Community 8 - "app.py"
Cohesion: 0.15
Nodes (18): artifact_selector(), build_knowledge(), _dataset_hint(), default_value_map(), discover(), feature_help_map(), main(), _range_midpoint() (+10 more)

### Community 9 - "test_agent_runner.py"
Cohesion: 0.20
Nodes (15): build_initial_state(), agent_runner.py  End-to-end runner for the explainability agent graph.  Asse, Assemble the initial agent state from config and inputs.      Args:         c, Run the compiled explainability graph end to end.      Args:         config_p, run_agent(), main(), _parse_sample(), main.py  Command-line entry point for the ML Explainability Agent.  Runs the ful (+7 more)

### Community 10 - "test_model_adapters.py"
Cohesion: 0.20
Nodes (15): GenericModelAdapter, Fallback adapter that matches any estimator.      This adapter emits the frame, Always match, acting as the universal fallback., Adapter for scikit-learn tree-based estimators.      Matches single decision t, Return whether the estimator is an sklearn tree or forest., SklearnTreeAdapter, _fit(), test_custom_adapter_takes_priority() (+7 more)

### Community 11 - "build_explanation_prompt"
Cohesion: 0.19
Nodes (16): build_explanation_prompt(), _feature_meanings(), explanation_prompt.py  Business-context-aware explanation prompt builder.  C, Render project-context sections when available., Build the business-context-aware explanation prompt.      Args:         state, Build a map of feature name to a human-readable meaning string.      Combines, Return the feature names referenced by the explanation signals.      Handles b, Render meanings for the features referenced in the explanation. (+8 more)

### Community 12 - "DataDictionaryLoader"
Cohesion: 0.18
Nodes (12): DataFrame, DataDictionaryLoader, data_dictionary_loader.py  Ingestion loader for the data dictionary artifact., Map a raw column header to its canonical key.          Args:             head, Return normalized feature records.          The dictionary is loaded automatic, Load and normalize a data dictionary into feature records.      Column headers, Load the data dictionary into a pandas DataFrame.          Returns:, test_load_csv() (+4 more)

### Community 13 - "Project Configuration"
Cohesion: 0.14
Nodes (16): Demo Configuration, Demo Explanation Settings, Demo Ingestion Settings, Demo Routing Settings, Project Configuration, Agent Framework Configuration, Data Paths Configuration, Explainability Methods Configuration (+8 more)

### Community 14 - "ModelAdapter"
Cohesion: 0.17
Nodes (9): ModelAdapter, Extract feature names from the fitted model when available., Extract class labels for classifiers when available., Extract the number of input features when available., Base class for framework-specific model adapters.      Subclasses declare the, Return whether this adapter can handle the given estimator.          Args:, Extract normalized metadata for the given model.          Args:             m, Infer whether the estimator is a classifier or regressor.          Returns: (+1 more)

### Community 15 - "decision_path_extractor.py"
Cohesion: 0.18
Nodes (7): test_extract_path(), tree_prediction_explainer.py  Converts decision paths into human-readable exp, Generate explanation         from decision path., TreePredictionExplainer, DecisionPathExtractor, decision_path_extractor.py  Extracts the exact path followed by a record insi, Extract decision path         for a single observation.

### Community 16 - "XGBoostAdapter"
Cohesion: 0.26
Nodes (12): default_registry(), Return a fresh registry configured with the built-in adapters., Adapter for XGBoost estimators.      Matches any object whose class is defined, Return whether the estimator comes from the xgboost package., XGBoostAdapter, _fitted_classifier(), _fitted_native_booster(), test_non_xgboost_model_does_not_match() (+4 more)

### Community 17 - "ConfigLoader"
Cohesion: 0.19
Nodes (7): test_load_config(), ConfigLoader, config_loader.py  Utility functions for loading project configuration files., Loads configuration from YAML files., Read YAML configuration file., Retrieve nested configuration values.          Example:         config.get("m, Return complete configuration.

### Community 18 - "test_ingestion_node.py"
Cohesion: 0.30
Nodes (10): ingestion_node(), Build the project knowledge object and populate agent state.      Reads the mo, ExplainabilityState, Shared state for the explainability agent graph.      ``config_path`` drives i, _make_model(), test_ingestion_node_handles_missing_artifacts(), test_ingestion_node_populates_knowledge_and_model(), test_ingestion_node_preserves_existing_state() (+2 more)

### Community 19 - "sklearn_tree.py"
Cohesion: 0.26
Nodes (6): base.py  Base model adapter interface.  Defines the :class:`ModelAdapter` co, generic.py  Generic fallback model adapter.  Used when no specialized framew, adapters  Model framework adapter (plugin) architecture.  Each adapter knows, registry.py  Adapter registry.  Resolves the correct :class:`ModelAdapter` f, sklearn_tree.py  Adapter for scikit-learn decision tree and forest models., xgboost_adapter.py  Adapter for XGBoost gradient-boosted models.  Handles bo

### Community 20 - "DecisionTreeReader"
Cohesion: 0.20
Nodes (5): test_tree_summary(), DecisionTreeReader, decision_tree_reader.py  Utilities for understanding and explaining Decision, Return basic tree information., Extract all split rules.

### Community 21 - "generate_demo_artifacts.py"
Cohesion: 0.25
Nodes (10): main(), generate_demo_artifacts.py  Create a self-contained set of demo artifacts for, Train and persist a fitted decision tree on the Iris dataset., Write a data dictionary describing the Iris features., Write a short project context markdown file., Write a demo configuration pointing at the generated artifacts., _train_model(), _write_config() (+2 more)

### Community 22 - "Enterprise ML Explainability Agent Vision"
Cohesion: 0.28
Nodes (9): Data Dictionary, Step-by-Step Development Roadmap, Enterprise ML Explainability Agent Vision, Internal Agent Knowledge Graph, Model Artifact, Project Context, Project Knowledge Object, Modular Repository Structure (+1 more)

### Community 23 - "Census Income (>50K) Classifier"
Cohesion: 0.25
Nodes (9): Census Income (>50K) Classifier, Census Income Target Variable (class), Census Income Key Drivers, Breast Tumour Malignancy Classifier, Breast Tumour Key Drivers, Breast Tumour Target Variable (benign vs malignant), California Median House Value Regressor, California Housing Key Drivers (+1 more)

### Community 24 - "AdapterRegistry"
Cohesion: 0.25
Nodes (5): AdapterRegistry, Resolve a model adapter from a loaded estimator.      Specialized adapters are, Args:             adapters: Optional iterable of adapter classes to use instead, Register a specialized adapter with priority over existing ones.          Args, Return an adapter instance for the given estimator.          Args:

### Community 25 - "._booster_details"
Cohesion: 0.29
Nodes (4): Extract normalized metadata plus booster-specific structure., Extract ensemble details (tree count) for the model., Return the underlying booster for sklearn-API or native models., Extract feature names carried on a native booster.

### Community 26 - "Explainability Engine"
Cohesion: 0.33
Nodes (6): DecisionPathExtractor, Explainability Engine, FeatureImportanceExtractor, Report Generator, SHAP Tool, Visualization Tool

### Community 27 - "Planner Agent"
Cohesion: 0.40
Nodes (5): ModelLoader, Planner Agent, Metrics Tool, Model Loader Tool, Prediction Tool

## Knowledge Gaps
- **45 isolated node(s):** `ml-explainability-agent`, `ConfigLoader`, `ModelLoader`, `DecisionTreeReader`, `DecisionPathExtractor` (+40 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **30 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ModelArtifactLoader` connect `ModelArtifactLoader` to `test_model_adapters.py`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `ModelAdapter` connect `ModelAdapter` to `XGBoostAdapter`, `test_model_adapters.py`, `sklearn_tree.py`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Why does `_read_text_file()` connect `ProjectContextLoader` to `ModelLoader`?**
  _High betweenness centrality (0.052) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `ModelAdapter` (e.g. with `GenericModelAdapter` and `SklearnTreeAdapter`) actually correct?**
  _`ModelAdapter` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `XGBoostAdapter` (e.g. with `AdapterRegistry` and `ModelAdapter`) actually correct?**
  _`XGBoostAdapter` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Explainer agent package.`, `Planner agent package.`, `Report agent package.` to the rest of the system?**
  _206 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `test_explainer_selector.py` be split into smaller, more focused modules?**
  _Cohesion score 0.061367621274108705 - nodes in this community are weakly interconnected._