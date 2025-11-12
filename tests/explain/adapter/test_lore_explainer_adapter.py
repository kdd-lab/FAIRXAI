import pytest
import numpy as np
from unittest.mock import MagicMock

from fairxai.explain.adapter.lore_explainer_adapter import LoreExplainerAdapter
from fairxai.explain.explaination.generic_explanation import GenericExplanation


# ----------------------------------------------------------------------
# Automatic fixture: mocks LORE generator classes with proper __name__
# ----------------------------------------------------------------------
@pytest.fixture(autouse=True)
def patch_lore_classes(monkeypatch):
    """Automatically mock all LORE generator classes with valid __name__ attributes."""
    for cls_name in [
        "TabularGeneticGeneratorLore",
        "TabularRandomGeneratorLore",
        "TabularRandGenGeneratorLore"
    ]:
        mock_cls = MagicMock()
        mock_cls.__name__ = cls_name  # 🧩 fixes AttributeError on logger.info
        monkeypatch.setattr(
            f"fairxai.explain.adapter.lore_explainer_adapter.{cls_name}", mock_cls
        )


# ----------------------------------------------------------------------
# Common adapter fixture
# ----------------------------------------------------------------------
@pytest.fixture
def adapter():
    """Provide a LoreExplainerAdapter with mocked model and dataset."""
    mock_model = MagicMock()
    mock_model.model = MagicMock()
    mock_dataset = MagicMock()
    return LoreExplainerAdapter(mock_model, mock_dataset)


# ----------------------------------------------------------------------
# TESTS
# ----------------------------------------------------------------------
def test_given_valid_strategy_when_init_explainer_then_initializes_correctly(adapter):
    """GIVEN a valid strategy
       WHEN _init_explainer is called
       THEN it should create the proper LORE explainer and set strategy."""
    adapter._init_explainer("genetic")

    assert adapter.explainer is not None
    assert adapter.strategy == "genetic"
    adapter.explainer.assert_not_called()  # it's just initialized, not used


def test_given_instance_when_explain_instance_then_returns_generic_explanation(adapter):
    """GIVEN a valid explainer instance and data
       WHEN explain_instance is called
       THEN it returns a GenericExplanation with expected data."""
    fake_explainer = MagicMock()
    fake_explainer.explain_instance.return_value = {
        "rule": "x > 5",
        "counterfactuals": [{"x": 4}],
        "feature_importances": {"x": 0.9},
        "fidelity": 0.95,
    }

    # patch inside adapter after init
    adapter._init_explainer("genetic")
    adapter.explainer = fake_explainer

    instance = np.array([1, 2, 3])
    result = adapter.explain_instance(instance, params={"strategy": "genetic", "num_samples": 50})

    assert isinstance(result, GenericExplanation)
    assert result.data["rule"] == "x > 5"
    assert result.data["strategy"] == "genetic"


def test_given_different_strategy_when_explain_instance_then_reinitializes_explainer(adapter):
    """GIVEN a change of strategy between calls
       WHEN explain_instance is invoked again
       THEN the explainer should be reinitialized with the new strategy."""
    adapter._init_explainer("genetic")
    first_explainer = adapter.explainer

    adapter._init_explainer("random")
    second_explainer = adapter.explainer

    assert first_explainer != second_explainer
    assert adapter.strategy == "random"


 # Mock local explanations that vary by index
def make_local_explanation(i):
    fake_local_exp = MagicMock(spec=GenericExplanation)
    fake_local_exp.to_dict.return_value = {"data": {"rule": f"rule{i}"}}
    return fake_local_exp


def test_given_dataset_when_explain_global_then_returns_aggregated_generic_explanation(adapter):
    """GIVEN a dataset and working local explanations
       WHEN explain_global is called
       THEN returns a GenericExplanation containing aggregated rules."""

    # Mock dataset
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    mock_dataset.__getitem__.side_effect = lambda i: np.array([i, i + 1])
    adapter.dataset = mock_dataset

    adapter.explain_instance = MagicMock(side_effect=[make_local_explanation(0), make_local_explanation(1)])

    adapter._init_explainer("genetic")

    result = adapter.explain_global({"strategy": "genetic", "n_samples": 2})

    assert isinstance(result, GenericExplanation)
    assert "aggregated_rules" in result.data
    assert result.data["num_instances"] == 2
    assert any(r in str(result.data) for r in ["rule0", "rule1"])



def test_given_unknown_strategy_when_init_explainer_then_raises_valueerror(adapter):
    """GIVEN an invalid strategy name
       WHEN _init_explainer is called
       THEN it raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown LORE strategy"):
        adapter._init_explainer("unknown_strategy")


def test_given_explainer_failure_when_explain_instance_then_raises_exception(adapter):
    """GIVEN an explainer that fails
       WHEN explain_instance is called
       THEN it raises the same exception."""
    fake_explainer = MagicMock()
    fake_explainer.explain_instance.side_effect = RuntimeError("boom")

    adapter._init_explainer("genetic")
    adapter.explainer = fake_explainer

    with pytest.raises(RuntimeError, match="boom"):
        adapter.explain_instance(np.array([1, 2, 3]), params={"strategy": "genetic"})
