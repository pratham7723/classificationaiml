import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay


def load_data():
    print("\n=== 1. Loading the Iris Dataset ===")
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    print(f"Features: {iris.feature_names}")
    print(f"Classes: {class_names}")
    print(f"Dataset shape: {X.shape}")
    return X, y, class_names


def preprocess_data(X, y):
    print("\n=== 2. Preprocessing: Train-Test Split and Scaling ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


def get_models():
    print("\n=== 3. Initializing Models ===")
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42)
    }
    print(f"Models: {list(models.keys())}")
    return models


def train_and_evaluate(models, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, class_names):
    print("\n=== 4. Training, Evaluation, and Confusion Matrices ===")
    accuracies = {}
    confusion_matrices = {}
    for name, model in models.items():
        print(f"\n--- {name} ---")
        # Use scaled data for KNN and Logistic Regression
        if name in ['KNN', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrices[name] = cm
        print(f"Accuracy: {acc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.title(f'{name} Confusion Matrix')
        plt.show()
    return accuracies, confusion_matrices


def print_summary_table(accuracies):
    print("\n=== 5. Model Accuracy Summary ===")
    print("+---------------------+----------+")
    print("| Model               | Accuracy |")
    print("+---------------------+----------+")
    for name, acc in accuracies.items():
        print(f"| {name:<19} | {acc:.4f}   |")
    print("+---------------------+----------+")


def plot_accuracy_bar_chart(accuracies):
    print("\n=== 6. Accuracy Bar Chart ===")
    plt.figure(figsize=(8, 5))
    plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Model Accuracy Comparison on Iris Dataset')
    for i, (name, acc) in enumerate(accuracies.items()):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center')
    plt.show()


def main():
    print("""
============================================
  Iris Dataset Classification Demo (AIML)
  Models: KNN, Decision Tree, Random Forest, Logistic Regression
  Steps: Preprocessing → Training → Evaluation → Comparison
============================================
    """)
    X, y, class_names = load_data()
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preprocess_data(X, y)
    models = get_models()
    accuracies, confusion_matrices = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, class_names
    )
    print_summary_table(accuracies)
    plot_accuracy_bar_chart(accuracies)
    print("\nDemo complete! You have seen the full classification workflow in action.")


if __name__ == "__main__":
    main() 