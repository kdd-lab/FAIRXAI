import pytest
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter
from fairxai.explain.explaination.generic_explanation import GenericExplanation


# ------------------------------------------------------------------------
# Helper subclass for testing purposes
# ------------------------------------------------------------------------
class ConcreteExplainer(GenericExplainerAdapter):
    """A minimal concrete subclass for testing abstract behavior."""
    explainer_name = "mock_explainer"
    supported_datasets = ["tabular"]
    supported_models = ["xgboost"]

    def explain_instance(self, instance):
        return {"instance_explained": instance}

    def explain_global(self):
        return {"global_explanation": True}


# ------------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------------

def test_given_abstract_class_when_instantiated_directly_then_raises_type_error():
    """GIVEN the abstract class GenericExplainerAdapter
       WHEN instantiated directly
       THEN Python should raise a TypeError."""
    with pytest.raises(TypeError):
        GenericExplainerAdapter(model="m", dataset="d")  # abstract -> cannot instantiate


def test_given_concrete_subclass_when_instantiated_then_creates_instance_correctly():
    """GIVEN a concrete subclass implementing all abstract methods
       WHEN instantiated
       THEN the object should be created with model and dataset attributes."""
    model = "dummy_model"
    dataset = "dummy_dataset"

    explainer = ConcreteExplainer(model=model, dataset=dataset)

    assert explainer.model == model
    assert explainer.dataset == dataset
    assert explainer.explainer_name == "mock_explainer"


def test_given_supported_types_when_checking_compatibility_then_returns_true():
    """GIVEN supported dataset and model types
       WHEN checking compatibility
       THEN returns True."""
    assert ConcreteExplainer.is_compatible("tabular", "xgboost")


def test_given_unsupported_types_when_checking_compatibility_then_returns_false():
    """GIVEN unsupported dataset and model types
       WHEN checking compatibility
       THEN returns False."""
    assert not ConcreteExplainer.is_compatible("image", "svm")


def test_given_wildcard_support_when_checking_compatibility_then_returns_true():
    """GIVEN a subclass that supports all datasets or models using wildcard
       WHEN checking compatibility
       THEN it should always return True."""
    class WildExplainer(ConcreteExplainer):
        supported_datasets = [GenericExplainerAdapter.WILDCARD]
        supported_models = [GenericExplainerAdapter.WILDCARD]

    assert WildExplainer.is_compatible("anything", "whatever")


def test_given_data_when_building_generic_explanation_then_returns_generic_explanation_instance():
    """GIVEN a valid data dictionary
       WHEN calling build_generic_explanation
       THEN it returns a GenericExplanation instance with correct metadata."""
    explainer = ConcreteExplainer(model="m", dataset="d")
    data = {"foo": "bar"}
    explanation = explainer.build_generic_explanation(data, explanation_type="local")

    assert isinstance(explanation, GenericExplanation)
    assert explanation.explainer_name == "mock_explainer"
    assert explanation.explanation_type == "local"
    assert explanation.data == data
