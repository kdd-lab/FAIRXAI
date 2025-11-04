import pandas as pd
import pytest

from fairxai.data.descriptor.tabular_descriptor import TabularDatasetDescriptor


def test_init_with_empty_columns():
    # Given
    data = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})

    # When
    descriptor = TabularDatasetDescriptor(data)

    # Then
    assert descriptor.data.equals(data)
    assert descriptor.categorical_columns == []
    assert descriptor.ordinal_columns == []


def test_init_with_defined_columns():
    # Given
    data = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})

    # When
    descriptor = TabularDatasetDescriptor(data, categorical_columns=['B'], ordinal_columns=['A'])

    # Then
    assert descriptor.categorical_columns == ['B']
    assert descriptor.ordinal_columns == ['A']


def test_describe_numeric_column_description():
    # Given
    data = pd.DataFrame({'A': [1, 2, 3]})
    descriptor = TabularDatasetDescriptor(data)

    # When
    result = descriptor.describe()

    # Then
    expected = {
        'numeric': {
            'A': {
                'index': 0,
                'min': 1,
                'max': 3,
                'mean': 2.0,
                'std': 1.0,
                'median': 2.0,
                'q1': 1.5,
                'q3': 2.5
            }
        },
        'categorical': {},
        'ordinal': {}
    }
    assert result == expected


def test_describe_categorical_column_description():
    # Given
    data = pd.DataFrame({'B': ['x', 'y', 'y', 'x', 'z']})
    descriptor = TabularDatasetDescriptor(data, categorical_columns=['B'])

    # When
    result = descriptor.describe()

    # Then
    expected = {
        'numeric': {},
        'categorical': {
            'B': {
                'index': 0,
                'distinct_values': ['x', 'y', 'z'],
                'count': {'x': 2, 'y': 2, 'z': 1}
            }
        },
        'ordinal': {}
    }
    assert result == expected


def test_describe_ordinal_column_description():
    # Given
    data = pd.DataFrame({'C': ['low', 'medium', 'high', 'low']})
    descriptor = TabularDatasetDescriptor(data, ordinal_columns=['C'])

    # When
    result = descriptor.describe()

    # Then
    expected = {
        'numeric': {},
        'categorical': {},
        'ordinal': {
            'C': {
                'index': 0,
                'distinct_values': ['low', 'medium', 'high'],
                'count': {'low': 2, 'medium': 1, 'high': 1}
            }
        }
    }
    assert result == expected


def test_get_numeric_columns():
    # Given
    data = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    descriptor = TabularDatasetDescriptor(data, categorical_columns=['B'])
    descriptor.describe()

    # When
    result = descriptor.get_numeric_columns()

    # Then
    assert result == ['A']


def test_get_ordinal_columns():
    # Given
    data = pd.DataFrame({'C': ['low', 'medium', 'high']})
    descriptor = TabularDatasetDescriptor(data, ordinal_columns=['C'])
    descriptor.describe()

    # When
    result = descriptor.get_ordinal_columns()

    # Then
    assert result == ['C']


def test_get_categorical_columns():
    # Given
    data = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
    descriptor = TabularDatasetDescriptor(data, categorical_columns=['B'])
    descriptor.describe()

    # When
    result = descriptor.get_categorical_columns()

    # Then
    assert result == ['B']


def test_invalid_column_type():
    # Given
    data = pd.DataFrame({'A': ['some', 'invalid', 'data']})
    descriptor = TabularDatasetDescriptor(data)

    # When / Then
    with pytest.raises(ValueError):
        descriptor.describe()
