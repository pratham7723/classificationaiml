import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import os

def run_classification_demo():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    class_names = iris.target_names
    feature_names = iris.feature_names
    # Sample of the dataset (first 5 rows)
    data_sample = [list(X[i]) + [class_names[y[i]]] for i in range(5)]
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'KNN': KNeighborsClassifier(),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=200, random_state=42)
    }
    accuracies = {}
    confusion_matrix_files = {}
    confusion_matrix_data = {}
    static_dir = 'static'
    os.makedirs(static_dir, exist_ok=True)
    predictions_table = []
    for name, model in models.items():
        if name in ['KNN', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        cm = confusion_matrix(y_test, y_pred)
        confusion_matrix_data[name] = cm.tolist()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot()
        plt.title(f'{name} Confusion Matrix')
        cm_filename = f'{static_dir}/cm_{name.replace(" ", "_").lower()}.png'
        plt.savefig(cm_filename)
        plt.close()
        confusion_matrix_files[name] = cm_filename
        # For KNN, save predictions table (first 10 test samples)
        if name == 'KNN':
            for i in range(min(10, len(y_test))):
                row = list(X_test[i]) + [class_names[y_test[i]], class_names[y_pred[i]]]
                predictions_table.append(row)
    # Accuracy bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Model Accuracy Comparison on Iris Dataset')
    for i, (name, acc) in enumerate(accuracies.items()):
        plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center')
    bar_chart_filename = f'{static_dir}/accuracy_bar_chart.png'
    plt.savefig(bar_chart_filename)
    plt.close()
    iris_description = (
        "The Iris dataset is a classic dataset in machine learning. "
        "It contains 150 samples of iris flowers, each described by four features: "
        "sepal length, sepal width, petal length, and petal width. "
        "The goal is to classify each sample into one of three species: Setosa, Versicolor, or Virginica."
    )
    return {
        'accuracies': accuracies,
        'confusion_matrix_files': confusion_matrix_files,
        'confusion_matrix_data': confusion_matrix_data,
        'bar_chart_file': bar_chart_filename,
        'data_sample': data_sample,
        'feature_names': feature_names,
        'class_names': class_names,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'predictions_table': predictions_table,
        'iris_description': iris_description
    } 