import pytest
from unittest.mock import MagicMock, patch
from fairxai.explain.explainer_manager.explainer_manager import ExplainerManager
from fairxai.explain.adapter.generic_explainer_adapter import GenericExplainerAdapter


# ============================================================
# GIVEN–WHEN–THEN TESTS
# ============================================================

def test_given_valid_dataset_and_model_when_initialize_then_loads_compatible_explainers(monkeypatch):
    """GIVEN a dataset/model pair
       WHEN ExplainerManager is initialized
       THEN it should load only compatible explainers."""
    # GIVEN mock explainer classes
    compatible_cls = MagicMock(spec=GenericExplainerAdapter)
    compatible_cls.__name__ = "CompatibleExplainer"
    compatible_cls.is_compatible.return_value = True

    incompatible_cls = MagicMock(spec=GenericExplainerAdapter)
    incompatible_cls.__name__ = "IncompatibleExplainer"
    incompatible_cls.is_compatible.return_value = False

    monkeypatch.setattr(
        ExplainerManager, "_get_all_explainer_classes", staticmethod(lambda: [compatible_cls, incompatible_cls])
    )

    # WHEN the manager is created
    manager = ExplainerManager(dataset_type="Tabular", model_name="MockModel")

    # THEN only the compatible explainer should be loaded
    explainers = manager.list_available_compatible_explainers()
    assert len(explainers) == 1
    assert explainers[0].__name__ == "CompatibleExplainer"


def test_given_exception_in_compatibility_check_when_load_explainers_then_logs_warning(monkeypatch):
    """GIVEN an explainer that raises during compatibility check
       WHEN ExplainerManager loads explainers
       THEN it should log a warning but continue."""
    broken_cls = MagicMock(spec=GenericExplainerAdapter)
    broken_cls.__name__ = "BrokenExplainer"
    broken_cls.is_compatible.side_effect = Exception("Boom")

    monkeypatch.setattr(
        ExplainerManager, "_get_all_explainer_classes", staticmethod(lambda: [broken_cls])
    )

    with patch("fairxai.explain.explainer_manager.explainer_manager.logger.warning") as mock_warn:
        manager = ExplainerManager("tabular", "modelx")

    # THEN a warning should be logged
    mock_warn.assert_called_once()
    # AND no explainers should be registered
    assert manager.explainers == {}


def test_given_available_explainers_when_create_explainer_then_returns_instance(monkeypatch):
    """GIVEN a manager with a compatible explainer
       WHEN create_explainer() is called with its name
       THEN it should return an explainer instance initialized with model and dataset."""
    explainer_cls = MagicMock(spec=GenericExplainerAdapter)
    explainer_cls.__name__ = "MyExplainer"
    explainer_cls.is_compatible.return_value = True
    explainer_instance = MagicMock()
    explainer_cls.return_value = explainer_instance

    monkeypatch.setattr(
        ExplainerManager, "_get_all_explainer_classes", staticmethod(lambda: [explainer_cls])
    )

    manager = ExplainerManager("tabular", "mock")
    model = MagicMock()
    dataset = MagicMock()

    instance = manager.create_explainer("MyExplainer", model, dataset)

    # THEN it should call the class constructor with provided model and dataset
    explainer_cls.assert_called_once_with(model=model, dataset=dataset)
    assert instance == explainer_instance


def test_given_invalid_explainer_name_when_create_explainer_then_raises_value_error(monkeypatch):
    """GIVEN a manager with one known explainer
       WHEN create_explainer() is called with an unknown name
       THEN it should raise a ValueError."""
    known_cls = MagicMock(spec=GenericExplainerAdapter)
    known_cls.__name__ = "KnownExplainer"
    known_cls.is_compatible.return_value = True

    monkeypatch.setattr(
        ExplainerManager, "_get_all_explainer_classes", staticmethod(lambda: [known_cls])
    )

    manager = ExplainerManager("tabular", "mock")

    with pytest.raises(ValueError, match="No compatible explainer named"):
        manager.create_explainer("UnknownExplainer", model_instance=None, dataset_instance=None)


def test_given_class_hierarchy_when_get_all_explainer_classes_then_returns_all_subclasses():
    """GIVEN a subclass hierarchy of GenericExplainerAdapter
       WHEN _get_all_explainer_classes() is called
       THEN it should return all subclasses recursively."""

    # Define minimal concrete subclasses that implement all abstract methods
    class MockBase(GenericExplainerAdapter):
        def explain_instance(self, instance=None):
            return "instance_explanation"

        def explain_global(self):
            return "global_explanation"

    class MockChild(MockBase):
        def explain_instance(self, instance=None):
            return "child_instance"

        def explain_global(self):
            return "child_global"

    class MockGrandChild(MockChild):
        def explain_instance(self, instance=None):
            return "grand_instance"

        def explain_global(self):
            return "grand_global"

    subclasses = ExplainerManager._get_all_explainer_classes()

    # THEN the returned list must include all subclasses
    names = [cls.__name__ for cls in subclasses]
    assert "MockBase" in names
    assert "MockChild" in names
    assert "MockGrandChild" in names



def test_given_mixed_case_names_when_normalize_type_names_then_returns_lowercase():
    """GIVEN mixed-case dataset/model names
       WHEN _normalize_type_names() is called
       THEN all names must be converted to lowercase."""
    result = ExplainerManager._normalize_type_names("Tabular", "MyModel")
    assert result == ("tabular", "mymodel")
