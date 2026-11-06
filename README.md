# Wine Quality Classification

## a. Problem Statement

Predict the sensory quality of Portuguese red Vinho Verde wine from eleven
physicochemical measurements. Compare five classifiers on exactly the same
held-out test set and demonstrate their predictions in an interactive Streamlit
application. This is **multiclass classification of the original quality score**,
not regression or an invented good/bad threshold.

## b. Project and Interactive Application

The application loads five saved pipelines. It never fits a model or preprocessing
step on uploaded data. The bundled test set is evaluated automatically on launch.

### Run Locally

Use Python **3.14** to match the tested environment. Run from this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On Windows, after installing dependencies, **double-click
[launch_app.bat](launch_app.bat)** to start Streamlit and open the interactive
frontend in the default browser. Alternatively, use the command above.
Streamlit prints the local URL, normally http://localhost:8501. An existing
server on that port may cause Streamlit to choose another port.

The app includes:

- CSV upload for labeled test data, accepting comma or semicolon delimiters.
- Selection of any of the five saved models.
- Accuracy, AUC, precision, recall, F1, and MCC for the selected model.
- A comparison table and chart for all five models on the active test data.
- Confusion matrix, optional row normalization, and classification report.
- Quality distribution, test data preview, and feature statistics.
- Downloads of the bundled test data, comparison table, and predictions with probabilities.

### Repository Layout

```text
.
|-- app.py
|-- evaluation.py
|-- train_models.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- launch_app.bat
|-- .streamlit/
|   `-- config.toml
|-- model/
|   |-- logistic_regression.joblib
|   |-- decision_tree.joblib
|   |-- knn.joblib
|   |-- naive_bayes.joblib
|   |-- support_vector_machine.joblib
|   |-- metadata.json
|   `-- metrics.csv
|-- tests/
|   |-- test_evaluation.py
|   |-- test_artifacts.py
|   `-- test_app.py
`-- wine+quality/
    |-- winequality-red.csv
    |-- winequality-white.csv
    `-- winequality.names
```

## c. Dataset Description

**Source:** UCI Wine Quality, supplied in [wine+quality](wine+quality).
The provided files contain 1,599 red-wine and 4,898 white-wine samples. This
experiment deliberately uses **only winequality-red.csv for all five models**;
the white-wine data is not mixed into training or testing.

There are eleven numeric input features:

| Feature | Meaning |
| --- | --- |
| fixed acidity | Fixed acid content |
| volatile acidity | Volatile acid content |
| citric acid | Citric acid concentration |
| residual sugar | Residual sugar concentration |
| chlorides | Chloride concentration |
| free sulfur dioxide | Free sulfur dioxide concentration |
| total sulfur dioxide | Total sulfur dioxide concentration |
| density | Wine density |
| pH | Acidity on the pH scale |
| sulphates | Sulphate concentration |
| alcohol | Alcohol content |

**Target:** `quality`, an expert sensory score on a 0-10 scale. The observed
red-wine classes are 3, 4, 5, 6, 7, and 8. No target binning is performed.
The supplied dataset has no missing values. Uploads with missing or nonnumeric
values are rejected rather than silently altered.

### Experimental Protocol

- Remove 240 exact duplicate rows **before splitting**, leaving 1,359 unique samples.
- Use one stratified 80/20 split with `random_state=42`: 1,087 training rows and 272 test rows.
- Export those exact held-out rows as [test_data.csv](test_data.csv), including the target.
- Fit standardization only on the training partition, inside each saved pipeline.
- Use the same feature order, target labels, and held-out rows for every classifier.
- Choose fixed model settings without tuning on test results; no test-set fitting or resampling.
- Report a single holdout experiment, not cross-validation estimates or confidence intervals.

| Quality | Unique Total | Training | Test |
| --- | ---: | ---: | ---: |
| 3 | 10 | 8 | 2 |
| 4 | 53 | 42 | 11 |
| 5 | 577 | 461 | 116 |
| 6 | 535 | 428 | 107 |
| 7 | 167 | 134 | 33 |
| 8 | 17 | 14 | 3 |

Classes are highly imbalanced. Rare-class estimates are particularly unstable
because quality 3 and quality 8 have only two and three test samples respectively.

## d. Models Used

| Model | Preprocessing and Settings |
| --- | --- |
| Logistic Regression | StandardScaler; LogisticRegression(max_iter=3000, random_state=42) |
| Decision Tree | No scaling; DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=42) |
| KNN | StandardScaler; KNeighborsClassifier(n_neighbors=11), uniform weights |
| Naive Bayes | StandardScaler; GaussianNB with default variance smoothing |
| Support Vector Machine | StandardScaler; SVC(C=3.0, kernel="rbf", probability=True, random_state=42) |

StandardScaler is saved inside the four applicable pipelines; no separate scaler
file or manual scaling is needed. The tree is also saved as a pipeline for a
consistent inference interface. SVM probability estimation uses its training
data only. Its probability calibration is internal to SVC; the uploaded test data
is never used for calibration. Scikit-learn 1.9 emits a deprecation warning for
`SVC(probability=True)`, but the pinned version supports it.

### Evaluation Metric Definitions

- **Accuracy:** fraction of samples with the correct quality label.
- **AUC:** unweighted macro one-vs-rest ROC AUC, computed from `predict_proba`,
  with columns aligned to the saved model's `classes_`.
- **Precision, Recall, F1:** unweighted macro averages over all six trained
  quality classes. Undefined per-class ratios use zero (`zero_division=0`).
- **MCC:** multiclass Matthews correlation coefficient, ranging from -1 to 1.

All six values below are scores, not percentages. Predictions use the model's
`predict` method; AUC uses its class probabilities. If an uploaded test set omits
any trained class, the app displays **N/A for AUC** with a warning instead of
inventing a score. Other macro metrics continue to include all six classes,
including absent classes. A one-class CSV is accepted, but its MCC can be zero
and its AUC is unavailable.

### Comparison Table

Measured on the bundled 272-row test set, rounded to four decimal places:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.5809 | 0.7821 | 0.2868 | 0.2673 | 0.2676 | 0.3129 |
| Decision Tree | 0.5184 | 0.6326 | 0.2494 | 0.2596 | 0.2543 | 0.2459 |
| KNN | 0.5735 | 0.7004 | 0.2776 | 0.2750 | 0.2740 | 0.3113 |
| Naive Bayes | 0.5441 | 0.6893 | 0.2922 | 0.3013 | 0.2948 | 0.3170 |
| Support Vector Machine | 0.5846 | 0.8121 | 0.2830 | 0.2764 | 0.2761 | 0.3292 |

Full-precision results are stored in [model/metrics.csv](model/metrics.csv) and
[model/metadata.json](model/metadata.json).

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy 0.5809 is close to SVM, and AUC 0.7821 is second highest. Macro F1 0.2676 is lower than KNN, Naive Bayes, and SVM, indicating that its aggregate accuracy does not translate to strong performance across all quality classes. |
| Decision Tree | Lowest accuracy (0.5184), AUC (0.6326), F1 (0.2543), and MCC (0.2459) for the chosen depth and leaf settings. Its axis-aligned partitions generalize less effectively than the other tested models on this split. |
| KNN | Accuracy 0.5735 is competitive with Logistic Regression. Its F1 0.2740 slightly exceeds Logistic Regression, but AUC 0.7004 trails both Logistic Regression and SVM. Distance-based neighborhoods still struggle with the sparse quality classes. |
| Naive Bayes | Highest macro precision (0.2922), recall (0.3013), and F1 (0.2948), despite lower accuracy (0.5441) and AUC (0.6893) than SVM. The overall class-balanced tradeoff is best among these fixed configurations, though correlated chemical inputs limit its independence assumption. |
| Support Vector Machine | Highest accuracy (0.5846), AUC (0.8121), and MCC (0.3292). Its nonlinear boundary performs well in aggregate, but macro F1 (0.2761) is below Naive Bayes. High AUC reflects ranking quality, not necessarily strong rare-class predictions. |
| Overall, Winner for your dataset? | **Naive Bayes by macro F1 (0.2948)**, the declared primary criterion because of class imbalance. **SVM wins accuracy, AUC, and MCC** and is preferable when those measures are the priority. Neither should be described as universally superior from this small single split. |

### Limitations

The most frequent class alone accounts for 116/272 (42.65%) of test samples.
All five models exceed that accuracy baseline, but macro F1 remains low and none
should be considered production-ready. No external validation, temporal split,
feature selection, or hyperparameter search was performed. Duplicate removal
prevents exact-row leakage but cannot establish that all remaining samples are
independent wines. Red-wine results do not establish white-wine performance.
The original target is ordinal; these classifiers treat its values as categories
and do not penalize errors according to their numeric distance.

## Reproduce the Experiment

Saved models are already included; training is **not required to launch the app**.
To regenerate all artifacts with the pinned environment:

```powershell
.\.venv\Scripts\python.exe train_models.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Training overwrites the five model pipelines, metadata, metrics, and test CSV.
Tests verify CSV validation, six-metric calculation, absent-class AUC handling,
duplicate-free train/test separation, artifact reload consistency, and Streamlit
model selection, normalization, upload handling, and displayed results.

The tested environment is Python 3.14.0, Streamlit 1.66.1, scikit-learn 1.9.0,
NumPy 2.5.2, pandas 3.0.5, Matplotlib 3.11.1, and joblib 1.6.0. Direct dependencies
are pinned in [requirements.txt](requirements.txt). Pickle/joblib models are
version-sensitive: use matching dependencies or retrain in the destination
environment. Only repository-owned saved models are loaded; model uploads are
not accepted because unpickling untrusted files can execute code.

## Test CSV Contract

Include exactly the named required fields below; extra columns are ignored.
Column order does not matter because the app selects features by name.

```csv
fixed acidity,volatile acidity,citric acid,residual sugar,chlorides,free sulfur dioxide,total sulfur dioxide,density,pH,sulphates,alcohol,quality
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4,5
```

The row above illustrates the schema, not a held-out experiment. Use the actual
[test_data.csv](test_data.csv) for the reported comparison. All fields must be
finite and numeric, quality must be an integer in the trained set `{3,4,5,6,7,8}`,
and the CSV must contain at least one row. Upload only independently held-out
test data. The app cannot determine whether arbitrary user-provided rows were
previously used for training. File uploads are limited to 10 MB and 10,000 rows.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload this project's contents. Include all
   five joblib files, metadata, source files, requirements, and test CSV. Do not
   upload the local virtual environment or secrets.
2. Visit https://streamlit.io/cloud and sign in with GitHub.
3. Choose **New app** / **Create app**, then select the repository and branch
   (usually `main`).
4. Set the main file path to `app.py`.
5. In advanced settings select **Python 3.14**, matching the tested environment.
   If that runtime is not offered, use a supported runtime in a separate local
   environment, install compatible dependency versions, retrain all five models,
   rerun tests, and commit matching requirements and artifacts before deploying.
6. Click **Deploy**, inspect build logs, and open the generated `.streamlit.app`
   URL. Verify bundled results, model selection, and a test CSV upload.
7. Add the public deployment URL to this README and your assignment submission.

**Deployment status:** implemented and tested locally; no public Community Cloud
deployment has been created. Publishing requires your GitHub repository and
authenticated Streamlit account. The localhost URL is not a public submission URL.

## Dataset Citation

P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis (2009).
*Modeling wine preferences by data mining from physicochemical properties.*
Decision Support Systems, 47(4), 547-553.
https://doi.org/10.1016/j.dss.2009.05.016

UCI dataset: https://archive.ics.uci.edu/dataset/186/wine+quality