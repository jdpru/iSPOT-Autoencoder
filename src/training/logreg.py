import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from src.configs import SEED

def train_and_evaluate_logreg(train_features, test_features, train_labels, 
                                        test_labels):
    print("=" * 50)

    # Train logistic regression
    lr_model = LogisticRegression(random_state=SEED, max_iter=1000)
    lr_model.fit(train_features, train_labels)
    
    # Predict on test set
    test_predictions = lr_model.predict(test_features)
    test_probabilities = lr_model.predict_proba(test_features)[:, 1]
    test_accuracy = accuracy_score(test_labels, test_predictions)
    roc_auc = roc_auc_score(test_labels, test_probabilities)
    
    print(f"\nLogistic Regression Results:")
    print(f"Test Accuracy: {test_accuracy:.3f}, ROC-AUC Score: {roc_auc:.3f}")
    
    return lr_model, roc_auc