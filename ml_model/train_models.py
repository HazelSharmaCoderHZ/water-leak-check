from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "data" / "location_aware_gis_leakage_dataset.csv"


def main():
    data = pd.read_csv(DATASET_PATH)
    if "Leakage_Flag" not in data.columns:
        raise ValueError(f"Dataset must contain a Leakage_Flag column: {DATASET_PATH}")

    features = data.drop(columns="Leakage_Flag")
    target = data["Leakage_Flag"]
    categorical_columns = features.select_dtypes(include=["object"]).columns.tolist()
    numeric_columns = features.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    features_train, _, target_train, _ = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    classifiers = {
        "random_forest_model.pkl": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
        "decision_tree_model.pkl": DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=50,
            min_samples_leaf=25,
            max_features=3,
            splitter="random",
            random_state=42,
        ),
        "logistic_regression_model.pkl": LogisticRegression(
            max_iter=100, C=0.01, solver="liblinear", random_state=42
        ),
        "svm_model.pkl": SVC(
            kernel="linear", C=0.01, probability=True, random_state=42
        ),
    }

    for filename, classifier in classifiers.items():
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("classifier", classifier),
            ]
        )
        model.fit(features_train, target_train)
        joblib.dump(model, ROOT_DIR / filename)
        print(f"Saved {ROOT_DIR / filename}")


if __name__ == "__main__":
    main()