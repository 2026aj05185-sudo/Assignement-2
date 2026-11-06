import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from evaluation import (
    FEATURES, METRIC_NAMES, MODEL_FILES, evaluate_model, read_csv, validate_data,
)


ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Wine Quality | Model Lab", layout="wide")


@st.cache_resource
def load_artifacts():
    metadata = json.loads((ROOT / "model" / "metadata.json").read_text(encoding="utf-8"))
    models = {
        name: joblib.load(ROOT / "model" / filename)
        for name, filename in MODEL_FILES.items()
    }
    return metadata, models


@st.cache_data(max_entries=8, show_spinner=False)
def score_data(content):
    metadata, models = load_artifacts()
    frame = read_csv(content)
    if len(frame) > 10000:
        raise ValueError("Please upload test data only, with at most 10,000 rows.")
    features, target = validate_data(frame, classes=metadata["classes"])
    results = {}
    for name, model in models.items():
        metrics, predictions, probabilities = evaluate_model(model, features, target)
        results[name] = {
            "metrics": metrics, "predictions": predictions, "probabilities": probabilities,
        }
    return features, target, results


try:
    metadata, models = load_artifacts()
except (FileNotFoundError, OSError, ValueError, ImportError) as error:
    st.error(f"Cannot load the trained models: {error}")
    st.code("python train_models.py", language="bash")
    st.stop()

with st.sidebar:
    st.markdown("### Wine Quality")
    st.caption("CLASSIFICATION LAB")
    st.divider()
    selected = st.selectbox("Model", list(MODEL_FILES), index=4)
    source = st.radio("Test dataset", ["Bundled holdout", "Upload CSV"])
    uploaded = st.file_uploader("Labeled test CSV", type=["csv"], disabled=source != "Upload CSV")
    st.download_button(
        "Download test data", (ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv", mime="text/csv", icon=":material/download:",
    )
    st.divider()
    st.markdown("**Experiment**")
    st.caption(f"Red wine | {metadata['train_rows']:,} training samples")
    st.caption("80/20 stratified split | seed 42")
    st.caption("Original quality classes | 3 through 8")
    st.caption("Precision, recall, F1: macro average")
    st.caption("AUC: macro one-vs-rest")

st.title("Wine Quality")
st.markdown("#### Classification benchmark")

if source == "Upload CSV" and uploaded is not None:
    content = uploaded.getvalue()
    dataset_name = uploaded.name
elif source == "Upload CSV":
    st.info("No test CSV selected.")
    st.stop()
else:
    content = (ROOT / "test_data.csv").read_bytes()
    dataset_name = "test_data.csv"

try:
    features, target, results = score_data(content)
except (ValueError, TypeError, UnicodeError, pd.errors.ParserError) as error:
    st.error(f"Invalid test CSV: {error}")
    st.stop()

st.caption(f"RED WINE / {dataset_name} / {len(features):,} samples / {len(FEATURES)} features")
selected_result = results[selected]
st.subheader(selected)
metric_columns = st.columns(6)
for column, name in zip(metric_columns, METRIC_NAMES):
    value = selected_result["metrics"][name]
    column.metric(name, "N/A" if value is None else f"{value:.4f}")
if selected_result["metrics"]["AUC"] is None:
    st.warning("AUC is unavailable: the test CSV must contain every trained quality class (3-8). "
               "The remaining metrics still use all six trained classes.")

comparison = pd.DataFrame([
    {"ML Model Name": name, **result["metrics"]} for name, result in results.items()
]).set_index("ML Model Name")
comparison_tab, detail_tab, data_tab = st.tabs(["Model comparison", "Model diagnostics", "Test data"])

with comparison_tab:
    st.subheader("Five models. One test set.")
    st.dataframe(comparison, width="stretch", column_config={
        name: st.column_config.NumberColumn(name, format="%.4f") for name in METRIC_NAMES
    })
    best_f1 = comparison["F1"].idxmax()
    best_accuracy = comparison["Accuracy"].idxmax()
    left, right = st.columns(2)
    left.markdown(f"**Best macro F1:** {best_f1} ({comparison.loc[best_f1, 'F1']:.4f})")
    right.markdown(f"**Best accuracy:** {best_accuracy} ({comparison.loc[best_accuracy, 'Accuracy']:.4f})")
    chart = comparison[["Accuracy", "F1"]].rename(columns={"F1": "Macro F1"})
    st.bar_chart(chart, horizontal=True, stack=False, color=["#297961", "#9b2448"], height=330)
    st.download_button(
        "Download comparison", comparison.to_csv().encode(), file_name="model_comparison.csv",
        mime="text/csv", icon=":material/download:",
    )

with detail_tab:
    matrix_column, report_column = st.columns([1, 1.2])
    labels = metadata["classes"]
    predictions = selected_result["predictions"]
    with matrix_column:
        st.subheader("Confusion matrix")
        normalize = st.toggle("Normalize by true class", value=False)
        matrix = confusion_matrix(target, predictions, labels=labels)
        display_matrix = matrix.astype(float)
        if normalize:
            totals = matrix.sum(axis=1, keepdims=True)
            display_matrix = np.divide(matrix, totals, out=np.zeros_like(display_matrix), where=totals != 0)
        figure, axis = plt.subplots(figsize=(6, 4.5))
        image = axis.imshow(display_matrix, cmap="PuRd", vmin=0)
        axis.set(xticks=range(len(labels)), yticks=range(len(labels)),
                 xticklabels=labels, yticklabels=labels, xlabel="Predicted quality", ylabel="True quality")
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = display_matrix[row, column]
                axis.text(column, row, f"{value:.2f}" if normalize else str(int(value)),
                          ha="center", va="center",
                          color="white" if value > display_matrix.max() / 2 else "#202b28")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)
    with report_column:
        st.subheader("Classification report")
        report = classification_report(
            target, predictions, labels=labels, output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), width="stretch")
        st.caption("Rare quality classes have very small support; per-class results are unstable.")
    output = features.copy()
    output["quality"] = target.to_numpy()
    output["predicted quality"] = predictions
    output["correct"] = target.to_numpy() == predictions
    for index, label in enumerate(labels):
        output[f"probability quality {label}"] = selected_result["probabilities"][:, index]
    st.download_button(
        "Download predictions", output.to_csv(index=False).encode(),
        file_name=f"{MODEL_FILES[selected].replace('.joblib', '')}_predictions.csv",
        mime="text/csv", icon=":material/download:",
    )
    st.dataframe(output[["quality", "predicted quality", "correct"]], width="stretch", height=240)

with data_tab:
    st.subheader("Quality distribution")
    counts = target.value_counts().reindex(metadata["classes"], fill_value=0).sort_index()
    counts.index = counts.index.astype(str)
    st.bar_chart(counts.rename("Samples"), color="#297961", height=250)
    st.subheader("Test samples")
    preview = features.copy()
    preview["quality"] = target.to_numpy()
    st.dataframe(preview, width="stretch", height=350)
    with st.expander("Feature summary"):
        st.dataframe(features.describe().round(4), width="stretch")

st.divider()
st.caption("UCI Wine Quality | Cortez et al. (2009) | Fixed trained pipelines; no training on uploaded data.")