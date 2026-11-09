import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    cross_val_predict,
    permutation_test_score,
)
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ----------------------------------------------------------------------
# 1. Load the data
# ----------------------------------------------------------------------
burst_table = pd.read_csv("duration_profiles.csv")
participant_table = pd.read_csv("ds007526/participants.tsv", sep="\t", na_values="n/a")

threshold_columns = [str(threshold) for threshold in range(20, 100, 5)]
threshold_values = np.array(range(20, 100, 5), dtype=float)

# Keep the resting condition, and only electrodes where a beta peak was found
burst_table = burst_table[(burst_table["task"] == "rest") & (burst_table["peak_found"] == True)].copy()

# ----------------------------------------------------------------------
# 2. Summarize each burst duration curve with intercept, slope, curvature
#    (fitted to the log of the durations, with the thresholds centered so
#    the intercept is the overall level at the middle of the curve)
# ----------------------------------------------------------------------
log_durations = np.log(burst_table[threshold_columns].values.astype(float))
centered_thresholds = threshold_values - threshold_values.mean()

# polyfit fits every curve at once when the curves are the columns
polynomial_coefficients = np.polyfit(centered_thresholds, log_durations.T, deg=2)

burst_table["curvature"] = polynomial_coefficients[0]
burst_table["slope"] = polynomial_coefficients[1]
burst_table["intercept"] = polynomial_coefficients[2]

# ----------------------------------------------------------------------
# 3. One row per subject, one column per electrode and curve feature
# ----------------------------------------------------------------------
wide_features = burst_table.pivot_table(
    index="subject",
    columns="electrode",
    values=["intercept", "slope", "curvature"],
    aggfunc="mean",
)
wide_features.columns = [f"{electrode}_{feature_name}" for feature_name, electrode in wide_features.columns]
feature_columns = list(wide_features.columns)

full_table = participant_table.merge(
    wide_features, left_on="participant_id", right_index=True, how="inner"
)
full_table = full_table.dropna(subset=["group", "age"]).reset_index(drop=True)

labels = (full_table["group"] == "PD").astype(int).values
feature_matrix = full_table[feature_columns].values
age_matrix = full_table[["age"]].values

print("Subjects:", len(full_table))
print("Parkinson's disease:", labels.sum(), " Healthy controls:", (labels == 0).sum())
print("Number of features:", len(feature_columns))

# ----------------------------------------------------------------------
# 4. Models: L2 logistic regression, regularization strength tuned in an inner loop
#    Imputation and scaling live inside the pipeline so nothing leaks from test folds
# ----------------------------------------------------------------------
inner_cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
outer_cross_validation = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
regularization_grid = {"classifier__C": np.logspace(-3, 1, 9)}

burst_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(l1_ratio=0, class_weight="balanced", max_iter=5000)),
])
burst_model = GridSearchCV(burst_pipeline, regularization_grid, cv=inner_cross_validation, scoring="roc_auc")

# Baseline: does age alone predict group?
age_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(l1_ratio=0, class_weight="balanced", max_iter=5000)),
])
age_model = GridSearchCV(age_pipeline, regularization_grid, cv=inner_cross_validation, scoring="roc_auc")

# ----------------------------------------------------------------------
# 5. Nested cross-validated performance (burst features vs age only)
# ----------------------------------------------------------------------
metric_names = ["roc_auc", "balanced_accuracy", "matthews_corrcoef"]

burst_results = cross_validate(burst_model, feature_matrix, labels, cv=outer_cross_validation,
                               scoring=metric_names, n_jobs=-1)
age_results = cross_validate(age_model, age_matrix, labels, cv=outer_cross_validation,
                             scoring=metric_names, n_jobs=-1)

print("\n=== Nested cross-validation (mean +/- standard deviation over 50 folds) ===")
for metric_name in metric_names:
    burst_scores = burst_results[f"test_{metric_name}"]
    age_scores = age_results[f"test_{metric_name}"]
    print(f"{metric_name:20s} burst features: {burst_scores.mean():.3f} +/- {burst_scores.std():.3f}"
          f"   age only: {age_scores.mean():.3f} +/- {age_scores.std():.3f}")

# ----------------------------------------------------------------------
# 6. Out-of-fold predictions, bootstrap confidence interval, confusion matrix, ROC curve
# ----------------------------------------------------------------------
prediction_cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
predicted_probabilities = cross_val_predict(
    burst_model, feature_matrix, labels, cv=prediction_cross_validation,
    method="predict_proba", n_jobs=-1,
)[:, 1]
predicted_labels = (predicted_probabilities >= 0.5).astype(int)

observed_auc = roc_auc_score(labels, predicted_probabilities)

random_generator = np.random.default_rng(0)
bootstrap_aucs = []
for bootstrap_index in range(2000):
    resampled_indices = random_generator.integers(0, len(labels), len(labels))
    if len(np.unique(labels[resampled_indices])) < 2:
        continue
    bootstrap_aucs.append(roc_auc_score(labels[resampled_indices], predicted_probabilities[resampled_indices]))
lower_bound, upper_bound = np.percentile(bootstrap_aucs, [2.5, 97.5])

print(f"\nOut-of-fold AUC: {observed_auc:.3f}  (95% bootstrap confidence interval {lower_bound:.3f} to {upper_bound:.3f})")
print("Confusion matrix (rows = true healthy control / Parkinson's, columns = predicted):")
print(confusion_matrix(labels, predicted_labels))

false_positive_rate, true_positive_rate, _ = roc_curve(labels, predicted_probabilities)
plt.figure(figsize=(5, 5))
plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {observed_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Healthy control vs Parkinson's disease (out-of-fold)")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=200)

# ----------------------------------------------------------------------
# 7. Permutation test: is the AUC better than chance?
#    (shuffles labels 500 times and repeats the whole nested procedure each time;
#     lower n_permutations if it runs too slowly)
# ----------------------------------------------------------------------
true_score, permuted_scores, permutation_p_value = permutation_test_score(
    burst_model, feature_matrix, labels,
    scoring="roc_auc",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=3),
    n_permutations=500, n_jobs=-1, random_state=0,
)
print(f"\nPermutation test: AUC = {true_score:.3f}, mean permuted AUC = {permuted_scores.mean():.3f}, "
      f"p = {permutation_p_value:.4f}")

# ----------------------------------------------------------------------
# 8. Which features drive the classifier? (fit on all data, for interpretation only)
# ----------------------------------------------------------------------
burst_model.fit(feature_matrix, labels)
final_coefficients = burst_model.best_estimator_.named_steps["classifier"].coef_[0]

coefficient_table = pd.DataFrame({"feature": feature_columns, "coefficient": final_coefficients})
coefficient_table["absolute_coefficient"] = coefficient_table["coefficient"].abs()
coefficient_table = coefficient_table.sort_values("absolute_coefficient", ascending=False)

print("\nBest regularization strength:", burst_model.best_params_["classifier__C"])
print("Top 15 features by absolute standardized coefficient:")
print(coefficient_table.head(15).to_string(index=False))
