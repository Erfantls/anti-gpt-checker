from typing import List, Tuple

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from ml.data_preparation import split_dataset, prepare_features
from ml.model_validation import calculate_classification_metrics


def evaluate_models(data_labels: List[Tuple[dict, int]]):
    train_data, train_labels, test_data, test_labels = split_dataset(data_labels, train_size=0.7)

    # Convert dictionary features to numpy arrays
    train_data = prepare_features(train_data)
    test_data = prepare_features(test_data)

    # Initialize models
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'MLP Classifier': MLPClassifier(max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Classifier': SVC(probability=True),
        'AdaBoost': AdaBoostClassifier(),
        'Gaussian Naive Bayes': GaussianNB(),
        'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis()
    }

    results = {}

    for name, model in models.items():
        # Train the model
        model.fit(train_data, train_labels)

        # Predict labels and probabilities
        y_pred = model.predict(test_data)
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(test_data)[:, 1]  # Probability estimates for the positive class
        else:
            y_scores = model.decision_function(test_data) if 'decision_function' in dir(model) else y_pred

        # Calculate metrics
        metrics = calculate_classification_metrics(test_labels, y_pred, y_scores)
        results[name] = metrics

    return results