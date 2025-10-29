import pytest
from fairxai.data.descriptor.text_descriptor import TextDatasetDescriptor


def test_describe_with_empty_data():
    # Given
    data = []
    descriptor = TextDatasetDescriptor(data)
    # When / Then
    with pytest.raises(ValueError, match="Empty dataset"):
        descriptor.describe()


def test_describe_with_raw_text_data():
    # Given
    data = ["This is a document.", "Another document here."]
    descriptor = TextDatasetDescriptor(data)
    # When
    result = descriptor.describe()
    # Then
    assert result["type"] == "text"
    assert result["n_documents"] == 2
    assert result["input_format"] == "raw_text"
    assert result["avg_length_words"] == pytest.approx(3.5)


def test_describe_with_dict_data():
    # Given
    data = [
        {"content": "Some content", "timestamp": "2025-10-29"},
        {"content": "Another piece of text"}
    ]
    descriptor = TextDatasetDescriptor(data)
    # When
    result = descriptor.describe()
    # Then
    assert result["type"] == "text"
    assert result["n_documents"] == 2
    assert result["input_format"] == "dict"
    assert "timestamp" in result["structure"]
    assert result["has_timestamp"] is True


def test_describe_with_dict_data_without_timestamp():
    # Given
    data = [
        {"content": "Data without timestamp", "meta": "metadata"},
        {"content": "Another content"}
    ]
    descriptor = TextDatasetDescriptor(data)
    # When
    result = descriptor.describe()
    # Then
    assert result["type"] == "text"
    assert result["n_documents"] == 2
    assert result["input_format"] == "dict"
    assert "timestamp" not in result["structure"]
    assert result["has_timestamp"] is False


def test_describe_with_unsupported_data_type():
    # Given
    data = [42, 3.14, {"content": "Valid dict"}]
    descriptor = TextDatasetDescriptor(data)
    # When / Then
    with pytest.raises(TypeError, match="Unsupported text format \\(use string or dict\\)"):
        descriptor.describe()
