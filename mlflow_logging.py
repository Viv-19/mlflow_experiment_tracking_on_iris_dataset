import pandas as pd
import mlflow
import mlflow.sklearn
import yaml

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load hyperparameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load dataset
df = pd.read_csv("data/iris_engineered.csv")
X = df.drop(columns=["target"])
y = df["target"]

# Split data using test_size from params.yaml
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["data_split"]["test_size"], random_state=42
)

# Set MLflow experiment
mlflow.set_experiment("Iris_Classification_Experiment")

# Mapping of model names to keys in params.yaml
param_keys = {
    "RandomForest": "RandomForest",
    "LogisticRegression": "LogisticRegression",
    "SVM": "SVM"
}

# Dictionary of models with their hyperparameters
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=params["RandomForest"]["n_estimators"],
        max_depth=params["RandomForest"]["max_depth"],
        random_state=42
    ),
    "LogisticRegression": LogisticRegression(
        max_iter=params["LogisticRegression"]["max_iter"]
    ),
    "SVM": SVC(
        kernel=params["SVM"]["kernel"],
        C=params["SVM"]["C"]
    )
}

# Loop and track each model
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')

        # Log parameters using mapping
        mlflow.log_params(params[param_keys[model_name]])

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Log model
        mlflow.sklearn.log_model(model, f"{model_name}_model")

        print(f"✅ {model_name} logged successfully.")
