import csv
from os import path

import pandas as pd
import pytest

from fairxai.data.dataset.tabular_dataset import TabularDataset


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'age': [25, 30, 22],
        'income': [50000, 60000, 45000],
        'gender': ['M', 'F', 'M'],
        'target': [1, 0, 1]
    })


def test_tabular_dataset_initialization(sample_data):
    # Given a DataFrame with columns 'age', 'income', 'gender', 'target'
    # When I initialize TabularDataset with target 'target' and categorical column 'gender'
    dataset = TabularDataset(
        data=sample_data,
        class_name='target',
        categorical_columns=['gender']
    )

    # Then
    assert dataset.class_name == 'target'
    # And categorical_columns should be set correctly
    assert dataset.categorical_columns == ['gender']
    # And target values should be extracted
    assert list(dataset.target) == [1, 0, 1]
    assert dataset.data.columns.tolist() == ['age', 'income', 'gender', 'target']


def test_initialization_without_class_name():
    # Given
    data = pd.DataFrame({
        "Feature1": [1, 2, 3],
        "Feature2": ["A", "B", "C"]
    })

    # When
    dataset = TabularDataset(data, categorical_columns=["Feature2"])

    # Then
    assert dataset.class_name is None
    assert dataset.target is None


def test_from_csv(tmp_path):
    # Given
    csv_content = [{"Feature1": 1, "Feature2": "A", "Target": 0},
                   {"Feature1": 2, "Feature2": "B", "Target": 1},
                   {"Feature1": 3, "Feature2": "C", "Target": 0}]

    csv_file_path = path.join(tmp_path, "test.csv")
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Feature1', 'Feature2', 'Target']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_content)

    # When
    dataset = TabularDataset.from_csv(csv_file_path, class_name="Target", categorical_columns=["Feature2"])

    # Then
    assert dataset.class_name == "Target"
    assert dataset.data.shape == (3, 3)
    assert list(dataset.data.columns) == ["Feature1", "Feature2", "Target"]


def test_from_dict_creates_dataset():
    # Given a dictionary representing tabular data
    data_dict = {
        'age': [20, 21],
        'income': [40000, 42000],
        'target': [0, 1]
    }

    # When I create a TabularDataset from the dictionary
    dataset = TabularDataset.from_dict(data_dict, class_name='target')

    # Then the dataset should have the correct target
    assert list(dataset.target) == [0, 1]
    # And the number of features should match the columns excluding the target
    assert dataset.get_number_of_features() == 2


def test_get_class_values_returns_unique_values(sample_data):
    # Given a TabularDataset with target column 'target'
    dataset = TabularDataset(sample_data, class_name='target', categorical_columns=['gender'])

    # When I call get_class_values
    class_values = dataset.get_class_values()

    # Then it should return unique target values
    assert set(class_values) == {0, 1}

def test_get_feature_names_and_index(sample_data):
    # Given a TabularDataset with numeric and categorical columns
    dataset = TabularDataset(sample_data, class_name='target', categorical_columns=['gender'])

    # When I get all feature names
    feature_names = dataset.get_feature_names()

    # Then the list should include all features except the target
    for name in ['age', 'income', 'gender']:
        assert name in feature_names

    # When I retrieve a feature by its index
    first_feature_name = dataset.get_feature_name(0)

    # Then it should return a valid feature name
    assert first_feature_name in feature_names

def test_get_class_values_raises_exception_if_no_class_name(sample_data):
    # Given a TabularDataset with no class_name
    dataset = TabularDataset(sample_data, categorical_columns=['gender'])

    # When I call get_class_values
    # Then it should raise an Exception
    with pytest.raises(Exception):
        dataset.get_class_values()


def test_get_feature_names_excludes_target(sample_data):
    # Given a dataset with numeric, categorical, and target columns
    dataset = TabularDataset(
        sample_data,
        class_name="target",
        categorical_columns=["gender"]
    )

    # When retrieving feature names without including the target
    feature_names = dataset.get_feature_names()

    # Then the target column should not appear in the list
    assert "target" not in feature_names
    assert set(feature_names) == {"age", "income", "gender"}


def test_get_feature_names_includes_target(sample_data):
    # Given a dataset with a defined target column
    dataset = TabularDataset(
        sample_data,
        class_name="target",
        categorical_columns=["gender"]
    )

    # When retrieving feature names including the target
    feature_names = dataset.get_feature_names(include_target=True)

    # Then the target column should appear in the list
    assert "target" in feature_names
    assert set(feature_names) == {"age", "income", "gender", "target"}

def test_get_number_of_features_excludes_target(sample_data):
    # Given a dataset with 3 features + 1 target
    dataset = TabularDataset(
        sample_data,
        class_name="target",
        categorical_columns=["gender"]
    )

    # When counting the number of features without including the target
    n_features = dataset.get_number_of_features()

    # Then the result should exclude the target column
    assert n_features == 3

def test_get_number_of_features_includes_target(sample_data):
    # Given the same dataset
    dataset = TabularDataset(
        sample_data,
        class_name="target",
        categorical_columns=["gender"]
    )

    # When counting the number of features including the target
    n_features_with_target = dataset.get_number_of_features(include_target=True)

    # Then the result should include all columns
    assert n_features_with_target == 4


def test_get_number_of_features():
    # Given
    data = pd.DataFrame({
        "Feature1": [1, 2, 3],
        "Feature2": ["A", "B", "C"],
        "Target": [0, 1, 0]
    })
    dataset = TabularDataset(data, class_name="Target", categorical_columns=["Feature2"])

    # When
    num_features = dataset.get_number_of_features()

    # Then
    assert num_features == 2


def test_get_feature_name_by_index():
    # Given
    data = pd.DataFrame({
        "Feature1": [1, 2, 3],
        "Feature2": ["A", "B", "C"]
    })
    dataset = TabularDataset(data, categorical_columns=["Feature2"])

    # When
    feature_name = dataset.get_feature_name(0)

    # Then
    assert feature_name == "Feature1"


def test_get_feature_name_invalid_index():
    # Given
    data = pd.DataFrame({
        "Feature1": [1, 2, 3],
        "Feature2": ["A", "B", "C"]
    })
    dataset = TabularDataset(data, categorical_columns=["Feature2"])

    # When / Then
    with pytest.raises(IndexError, match="No feature found with index -1"):
        dataset.get_feature_name(-1)
