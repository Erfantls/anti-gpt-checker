from typing import List, Tuple, Dict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import SelectKBest
import pandas as pd


def find_significant_features(data):
    """
    Identifies the most significant features in a list of data.

    Args:
    data (list of tuples): A list where each element is a tuple containing a dictionary of features and a label.
    num_features (int): The number of top features to select.

    Returns:
    list: Names of the top `num_features` significant features.
    """
    # Convert list of tuples into a DataFrame
    df = pd.DataFrame([item[0] for item in data])
    labels = [item[1] for item in data]

    # Fill missing values if necessary
    df.fillna(method='ffill', inplace=True)

    # Compute mutual information
    mi_scores = mutual_info_classif(df, labels)

    # Create a Series with feature names as the index and MI scores as the values
    mi_series = pd.Series(mi_scores, index=df.columns)

    # Sort the features by their mutual information scores in descending order
    sorted_features = mi_series.sort_values(ascending=False).index.tolist()

    return sorted_features


def split_dataset(data_labels: List[Tuple[Dict, int]], train_size: float):
    """
    Shuffles and splits the data into training and testing sets.
    """
    np.random.shuffle(data_labels)
    split_idx = int(len(data_labels) * train_size)
    train_paired, test_paired = data_labels[:split_idx], data_labels[split_idx:]

    # Extract data and labels
    train_data = [x[0] for x in train_paired]
    train_labels = [x[1] for x in train_paired]
    test_data = [x[0] for x in test_paired]
    test_labels = [x[1] for x in test_paired]

    return train_data, train_labels, test_data, test_labels

def prepare_features(data: List[Dict]):
    """
    Converts list of dictionaries into a matrix of features suitable for sklearn models.
    """
    vectorizer = DictVectorizer(sparse=False)
    return vectorizer.fit_transform(data)