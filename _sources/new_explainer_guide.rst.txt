Guide to integrating an explanation method in FAIRXAI
=====================================================

This guide describes the steps required to integrate a new explanation method (explainer)
into the FAIRXAI framework.

.. contents::
   :local:
   :depth: 2

---

1. Architecture Overview
------------------------

The FAIRXAI framework uses an **Adapter-based** system.
Each explanation method must be encapsulated in a class that acts as a bridge between the specific library
(e.g., SHAP, LIME, Grad-CAM) and the project’s standard interface.

The **auto-discovery** system automatically loads all adapters located in the
``fairxai.explain.adapter`` package.

---

2. Implementing the Adapter
---------------------------

To add a new explainer, create a new Python file in the
``fairxai/explain/adapter/`` directory (e.g., ``my_explainer_adapter.py``).

Step A: Inheritance and Metadata
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The class must inherit from :class:`~fairxai.explain.adapter.GenericExplainerAdapter`
and define three key attributes:

.. attribute:: explainer_name
   :type: str

   Unique identifier used in YAML configuration files.

.. attribute:: supported_datasets
   :type: List[str]

   List of supported dataset types (e.g., ``["image"]``, ``["tabular"]``, or ``["*"]`` for all).

.. attribute:: supported_models
   :type: List[str]

   List of supported model types (matching model class names, e.g., ``["Conv2d"]``, ``["RandomForestClassifier"]``, or ``["*"]``).

Step B: Required Methods
~~~~~~~~~~~~~~~~~~~~~~~~

You must implement the following methods:

.. method:: __init__(self, model, dataset)
   :noindex:

   Initializes the explainer with the model (BBox wrapper) and the dataset.

.. method:: explain_instance(self, instance, params=None)
   :noindex:

   Generates a **local** explanation for a single instance.

   :param instance: The data instance to explain.
   :param params: Optional parameters dictionary.
   :return: A list of :class:`~fairxai.explain.explaination.GenericExplanation` objects.

.. method:: explain_global(self)
   :noindex:

   Generates a **global** explanation for the model.
   If unsupported, raise ``NotImplementedError``.

Code Example
~~~~~~~~~~~~

.. code-block:: python

    from typing import List, Optional, Dict, Any
    from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
    from fairxai.explain.explaination.feature_importance_explanation import FeatureImportanceExplanation

    class MyExplainerAdapter(GenericExplainerAdapter):
        explainer_name: str = "my_method"
        supported_datasets: List[str] = ["tabular"]
        supported_models: List[str] = ["*"]

        def __init__(self, model, dataset):
            super().__init__(model, dataset)
            # Specific initialization (e.g., loading external library)

        def explain_instance(self, instance, params: Optional[Dict[str, Any]] = None) -> List[FeatureImportanceExplanation]:
            # 1. Explanation computation logic
            # 2. Format importance data (e.g., { "feature_1": 0.8, "feature_2": 0.1 })
            importances = {"age": 0.7, "income": 0.3}

            # 3. Return a standard Explanation object
            return [
                FeatureImportanceExplanation(
                    explainer_name=self.explainer_name,
                    data=importances,
                    global_scope=False
                )
            ]

        def explain_global(self):
            raise NotImplementedError("Global explanation not supported.")

---

3. Handling Results (Explanation)
---------------------------------

The framework expects the ``explain_*`` methods to return a list of objects inheriting from
:class:`~fairxai.explain.explaination.GenericExplanation`.

Available subclasses
~~~~~~~~~~~~~~~~~~~~

- :class:`~fairxai.explain.explaination.FeatureImportanceExplanation`
  For explanations based on feature importance scores (pixels, tabular columns, etc.).
  Optionally supports a ``visualization`` dictionary for visual assets (e.g., Grad-CAM overlays).

- :class:`~fairxai.explain.explaination.RuleExplanation`
  For rule-based explanations (e.g., SHAP, LORE).

- :class:`~fairxai.explain.explaination.CounterfactualExplanation`
  For counterfactual-based explanations.

If your method produces a radically different output, you should create a new subclass of
:class:`~fairxai.explain.explaination.GenericExplanation` in
``fairxai/explain/explaination/``.

---

4. Registration and Verification
--------------------------------

Manual class registration is **not required**. Thanks to the dynamic discovery mechanism:

1. Place your file in ``fairxai/explain/adapter/``.
2. When the ``Project`` starts, the framework will automatically load your adapter.
3. You can verify the loading by checking the logs:

   .. code-block:: text

      INFO: Found X compatible explainers.

---

5. Usage via Configuration
--------------------------

Once integrated, the new explainer can be invoked through a YAML pipeline file:

.. code-block:: yaml

    pipeline:
      - explainer: "my_method"
        mode: "local"
        params:
          instance_index: 0

---

Developer Checklist
-------------------

- [ ] Does the class inherit from ``GenericExplainerAdapter``?
- [ ] Is ``explainer_name`` unique and descriptive?
- [ ] Are ``supported_datasets`` and ``supported_models`` correctly set?
- [ ] Is the file saved in the ``adapter`` folder?
- [ ] Are the results encapsulated in ``GenericExplanation`` (or derived) objects?
