import json
import platform
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import MODEL_FILES, TARGET, evaluate_model, read_csv, validate_data


ROOT = Path(__file__).resolve().parent
SEED = 42
DATA_URL = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"

def load_dataset(dataset_path=None):
    local_workbook = ROOT / "winequality-red.csv"
    if dataset_path is None and local_workbook.exists():
        dataset_path = local_workbook
    if dataset_path:
        return pd.read_excel(dataset_path, header=1, engine="xlrd")
    with urlopen(DATA_URL, timeout=90) as response:
        archive_bytes = response.read()
    with ZipFile(io.BytesIO(archive_bytes)) as archive:
        filename = next(name for name in archive.namelist() if name.endswith(".xls"))
        return pd.read_excel(io.BytesIO(archive.read(filename)), header=1, engine="xlrd")

def train():
    source = ROOT / "wine+quality" / "winequality-red.csv"
    original = read_csv(source)
    dataset = original.drop_duplicates().reset_index(drop=True)
    features, target = validate_data(dataset)
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=0.2, random_state=SEED, stratify=target
    )
    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=3000, random_state=SEED)
        ),
        "Decision Tree": make_pipeline(
            DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=SEED)
        ),
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=11)),
        "Naive Bayes": make_pipeline(StandardScaler(), GaussianNB()),
        "Support Vector Machine": make_pipeline(
            StandardScaler(), SVC(C=3.0, kernel="rbf", probability=True, random_state=SEED)
        ),
    }
    output = ROOT / "model"
    output.mkdir(exist_ok=True)
    test_data = test_features.copy()
    test_data[TARGET] = test_target
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    results = []
    for name, model in models.items():
        model.fit(train_features, train_target)
        metrics, _, _ = evaluate_model(model, test_features, test_target)
        joblib.dump(model, output / MODEL_FILES[name])
        results.append({"ML Model Name": name, **metrics})
        print(name, json.dumps(metrics), flush=True)
    comparison = pd.DataFrame(results)
    comparison.to_csv(output / "metrics.csv", index=False)
    summary = {
        "dataset": "UCI Wine Quality - red wine",
        "original_rows": len(original),
        "duplicates_removed": len(original) - len(dataset),
        "unique_rows": len(dataset),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "features": list(features.columns),
        "classes": sorted(map(int, target.unique())),
        "class_counts": {str(label): int(count) for label, count in target.value_counts().sort_index().items()},
        "train_class_counts": {str(label): int(count) for label, count in train_target.value_counts().sort_index().items()},
        "test_class_counts": {str(label): int(count) for label, count in test_target.value_counts().sort_index().items()},
        "seed": SEED,
        "test_fraction": 0.2,
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "winner_by_macro_f1": comparison.loc[comparison["F1"].idxmax(), "ML Model Name"],
        "metrics": results,
    }
    (output / "metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved five pipelines and {len(test_data)} held-out rows. Winner by macro F1: "
          f"{summary['winner_by_macro_f1']}")


if __name__ == "__main__":
    train()